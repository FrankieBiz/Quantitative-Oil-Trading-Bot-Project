"""
Order Execution module for Interactive Brokers via ib_insync.

Handles connection management, bracket order submission, order lifecycle
tracking, portfolio queries, and event-driven fill/cancel/error handling.
All orders are logged to the database and throttled to respect IBKR rate limits.
"""

import asyncio
import time
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ib_insync import IB, Contract, Future, Stock, Order as IBOrder, Trade, LimitOrder, MarketOrder, StopOrder
from loguru import logger

from config import settings
from db.models import (
    Order,
    Trade as DBTrade,
    TradeDirection,
    TradeStatus,
    OrderType,
    get_session,
)


class BrokerExecutor:
    """Manages the full order lifecycle against Interactive Brokers.

    Responsibilities:
        - Maintain a resilient IB Gateway / TWS connection with heartbeat
          monitoring and exponential-backoff reconnection.
        - Submit bracket orders (market entry + limit take-profit + stop loss).
        - Enforce an order-throttle of ``ORDER_THROTTLE_MAX_PER_MINUTE`` per
          rolling minute.
        - Provide helpers for order modification, cancellation, trailing stops,
          and portfolio queries.
        - Persist every order and fill to the database via SQLAlchemy models.
        - Apply a configurable slippage model for back-testing consistency.

    Attributes:
        ib: The underlying ``ib_insync.IB`` connection instance.
        connected: Whether the broker connection is currently active.
    """

    # --------------------------------------------------------------------- #
    #  Initialisation
    # --------------------------------------------------------------------- #

    def __init__(self) -> None:
        """Initialise the BrokerExecutor with default state."""
        self.ib: IB = IB()
        self.connected: bool = False
        self._reconnect_attempts: int = 0
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._order_timestamps: deque = deque()
        self._active_brackets: Dict[str, Dict] = {}
        self._slippage_pct: float = settings.SLIPPAGE_PCT

        # Wire up ib_insync event callbacks
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.errorEvent += self.on_error
        self.ib.disconnectedEvent += self.on_disconnected

    # --------------------------------------------------------------------- #
    #  Connection Management
    # --------------------------------------------------------------------- #

    async def connect(self) -> bool:
        """Connect to IBKR Gateway / TWS.

        Uses paper-trading port 7497 by default.  When ``LIVE_MODE`` is
        ``True`` in settings, port 7496 is used instead.

        Returns:
            ``True`` if the connection succeeded, ``False`` otherwise.
        """
        port = settings.IBKR_PORT
        mode = "LIVE" if settings.LIVE_MODE else "PAPER"
        logger.info(
            "Connecting to IBKR | host={} port={} client_id={} mode={}",
            settings.IBKR_HOST,
            port,
            settings.IBKR_CLIENT_ID,
            mode,
        )

        try:
            await self.ib.connectAsync(
                host=settings.IBKR_HOST,
                port=port,
                clientId=settings.IBKR_CLIENT_ID,
                timeout=settings.IBKR_TIMEOUT,
            )
            self.connected = True
            self._reconnect_attempts = 0
            logger.success("Connected to IBKR ({} mode)", mode)
            self._start_heartbeat()
            return True
        except Exception as exc:
            logger.error("IBKR connection failed: {}", exc)
            self.connected = False
            return False

    async def reconnect(self) -> bool:
        """Reconnect with exponential back-off.

        Retries up to ``IBKR_MAX_RECONNECT_ATTEMPTS`` times.  The delay
        between attempts doubles each time, starting at
        ``IBKR_RECONNECT_DELAY`` seconds, capped at 120 s.

        Returns:
            ``True`` once reconnected, ``False`` if all attempts are exhausted.
        """
        self._stop_heartbeat()

        while self._reconnect_attempts < settings.IBKR_MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1
            delay = min(
                settings.IBKR_RECONNECT_DELAY * (2 ** (self._reconnect_attempts - 1)),
                120.0,
            )
            logger.warning(
                "Reconnect attempt {}/{} in {:.1f}s",
                self._reconnect_attempts,
                settings.IBKR_MAX_RECONNECT_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)

            try:
                if self.ib.isConnected():
                    self.ib.disconnect()
            except Exception:
                pass

            success = await self.connect()
            if success:
                return True

        logger.critical(
            "Failed to reconnect after {} attempts",
            settings.IBKR_MAX_RECONNECT_ATTEMPTS,
        )
        return False

    async def disconnect(self) -> None:
        """Gracefully disconnect from IBKR and stop the heartbeat."""
        self._stop_heartbeat()
        if self.ib.isConnected():
            self.ib.disconnect()
        self.connected = False
        logger.info("Disconnected from IBKR")

    # --------------------------------------------------------------------- #
    #  Heartbeat
    # --------------------------------------------------------------------- #

    def _start_heartbeat(self) -> None:
        """Start the background heartbeat coroutine."""
        self._stop_heartbeat()
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

    def _stop_heartbeat(self) -> None:
        """Cancel the running heartbeat task if one exists."""
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Periodically check connection health and trigger reconnection.

        Runs every ``IBKR_HEARTBEAT_INTERVAL`` seconds.  If the connection
        is found to be dead, ``reconnect()`` is invoked automatically.
        """
        while True:
            try:
                await asyncio.sleep(settings.IBKR_HEARTBEAT_INTERVAL)
                if not self.ib.isConnected():
                    logger.warning("Heartbeat detected disconnection")
                    self.connected = False
                    await self.reconnect()
                else:
                    # Lightweight request to confirm API responsiveness
                    self.ib.reqCurrentTime()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Heartbeat error: {}", exc)

    # --------------------------------------------------------------------- #
    #  Order Throttle
    # --------------------------------------------------------------------- #

    def _check_throttle(self) -> bool:
        """Enforce the per-minute order throttle.

        Returns:
            ``True`` if the order may proceed, ``False`` if the throttle
            limit has been reached.
        """
        now = time.monotonic()
        # Purge timestamps older than 60 seconds
        while self._order_timestamps and (now - self._order_timestamps[0]) > 60.0:
            self._order_timestamps.popleft()

        if len(self._order_timestamps) >= settings.ORDER_THROTTLE_MAX_PER_MINUTE:
            return False
        return True

    def _record_order_timestamp(self) -> None:
        """Record the current monotonic time for throttle tracking."""
        self._order_timestamps.append(time.monotonic())

    async def _wait_for_throttle(self) -> None:
        """Block until the throttle window has capacity.

        Sleeps in 0.5-second increments until the oldest tracked timestamp
        falls outside the rolling 60-second window.
        """
        while not self._check_throttle():
            wait_time = 60.0 - (time.monotonic() - self._order_timestamps[0]) + 0.1
            logger.debug("Order throttle active, waiting {:.1f}s", wait_time)
            await asyncio.sleep(min(wait_time, 0.5))

    # --------------------------------------------------------------------- #
    #  Contract Helpers
    # --------------------------------------------------------------------- #

    def _create_contract(self, instrument: str) -> Contract:
        """Build the appropriate ib_insync Contract for a given instrument.

        Supports the following symbols:
            - **CL** / **BZ** -- front-month NYMEX futures.
            - **USO** / **XLE** -- ARCA-listed equities / ETFs.

        Args:
            instrument: Ticker symbol (e.g. ``"CL"``, ``"USO"``).

        Returns:
            A fully-qualified ``ib_insync.Contract``.

        Raises:
            ValueError: If the instrument is not recognised.
        """
        if instrument in settings.FUTURES_INSTRUMENTS:
            spec = settings.FUTURES_INSTRUMENTS[instrument]
            # Front-month: empty lastTradeDateOrContractMonth lets IBKR resolve
            # the continuous front contract when ``includeExpired=False``.
            contract = Future(
                symbol=instrument,
                exchange=spec["exchange"],
                currency=spec["currency"],
            )
            # Resolve to front-month via qualification
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                return qualified[0]
            # Fallback: return unqualified contract
            logger.warning(
                "Could not qualify futures contract for {}; using unqualified",
                instrument,
            )
            return contract

        if instrument in settings.ETF_INSTRUMENTS:
            spec = settings.ETF_INSTRUMENTS[instrument]
            contract = Stock(
                symbol=instrument,
                exchange="SMART",
                currency=spec["currency"],
                primaryExchange=spec["exchange"],
            )
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                return qualified[0]
            logger.warning(
                "Could not qualify stock contract for {}; using unqualified",
                instrument,
            )
            return contract

        raise ValueError(f"Unknown instrument: {instrument}")

    # --------------------------------------------------------------------- #
    #  Order Submission
    # --------------------------------------------------------------------- #

    async def submit_bracket_order(
        self,
        instrument: str,
        direction: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[str]:
        """Submit a bracket order: market entry with attached stop-loss and take-profit.

        The entry leg uses a **market** order for immediate fill.  The
        take-profit leg uses a **limit** order offset by ±0.05 % from the
        target price (``LIMIT_OFFSET_PCT``) to improve fill probability.
        The stop-loss leg uses a **stop** order at the specified price.

        Args:
            instrument: Ticker symbol.
            direction: ``"LONG"`` or ``"SHORT"``.
            quantity: Number of contracts / shares.
            entry_price: Reference entry price (used for logging and slippage).
            stop_loss: Stop-loss trigger price.
            take_profit: Take-profit limit price.

        Returns:
            A ``trade_id`` string that groups the three legs, or ``None`` if
            the order could not be submitted.
        """
        if not self.connected or not self.ib.isConnected():
            logger.error("Cannot submit order -- not connected to IBKR")
            return None

        await self._wait_for_throttle()

        trade_id = f"T-{uuid.uuid4().hex[:12]}"
        action = "BUY" if direction == "LONG" else "SELL"
        reverse_action = "SELL" if direction == "LONG" else "BUY"

        contract = self._create_contract(instrument)

        # -- Compute limit offsets for exit legs -------------------------
        offset = settings.LIMIT_OFFSET_PCT
        if direction == "LONG":
            tp_limit = round(take_profit * (1 - offset), 2)
            sl_price = round(stop_loss, 2)
        else:
            tp_limit = round(take_profit * (1 + offset), 2)
            sl_price = round(stop_loss, 2)

        # -- Build IB bracket order using ib_insync helper ---------------
        parent = MarketOrder(action, quantity)
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False

        take_profit_order = LimitOrder(reverse_action, quantity, tp_limit)
        take_profit_order.orderId = self.ib.client.getReqId()
        take_profit_order.parentId = parent.orderId
        take_profit_order.transmit = False

        stop_loss_order = StopOrder(reverse_action, quantity, sl_price)
        stop_loss_order.orderId = self.ib.client.getReqId()
        stop_loss_order.parentId = parent.orderId
        stop_loss_order.transmit = True  # last child transmits the group

        try:
            parent_trade = self.ib.placeOrder(contract, parent)
            tp_trade = self.ib.placeOrder(contract, take_profit_order)
            sl_trade = self.ib.placeOrder(contract, stop_loss_order)

            self._record_order_timestamp()
            self._record_order_timestamp()
            self._record_order_timestamp()

            self._active_brackets[trade_id] = {
                "parent_trade": parent_trade,
                "tp_trade": tp_trade,
                "sl_trade": sl_trade,
                "instrument": instrument,
                "direction": direction,
                "quantity": quantity,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

            # Persist to DB
            self._log_order_to_db(
                order_id=str(parent.orderId),
                trade_id=trade_id,
                instrument=instrument,
                order_type=OrderType.MARKET,
                side=action,
                quantity=quantity,
                limit_price=None,
                stop_price=None,
            )
            self._log_order_to_db(
                order_id=str(take_profit_order.orderId),
                trade_id=trade_id,
                instrument=instrument,
                order_type=OrderType.LIMIT,
                side=reverse_action,
                quantity=quantity,
                limit_price=tp_limit,
                stop_price=None,
            )
            self._log_order_to_db(
                order_id=str(stop_loss_order.orderId),
                trade_id=trade_id,
                instrument=instrument,
                order_type=OrderType.STOP,
                side=reverse_action,
                quantity=quantity,
                limit_price=None,
                stop_price=sl_price,
            )

            # Create Trade record
            self._create_trade_record(
                trade_id=trade_id,
                instrument=instrument,
                direction=direction,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            logger.info(
                "Bracket order submitted | trade_id={} instrument={} direction={} "
                "qty={} entry={} SL={} TP={}",
                trade_id,
                instrument,
                direction,
                quantity,
                entry_price,
                stop_loss,
                take_profit,
            )
            return trade_id

        except Exception as exc:
            logger.error("Failed to submit bracket order: {}", exc)
            return None

    # --------------------------------------------------------------------- #
    #  Order Management
    # --------------------------------------------------------------------- #

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a live order by its IBKR order ID.

        Args:
            order_id: The string representation of the IB order ID.

        Returns:
            ``True`` if the cancellation request was sent, ``False`` on error.
        """
        try:
            int_id = int(order_id)
        except ValueError:
            logger.error("Invalid order_id for cancellation: {}", order_id)
            return False

        for trade in self.ib.openTrades():
            if trade.order.orderId == int_id:
                self.ib.cancelOrder(trade.order)
                logger.info("Cancel requested for order_id={}", order_id)
                self._update_order_status_db(order_id, "CANCEL_REQUESTED")
                return True

        logger.warning("Order {} not found among open trades", order_id)
        return False

    def modify_order(self, order_id: str, new_price: float) -> bool:
        """Modify the limit or stop price of an existing order.

        For limit orders the limit price is updated; for stop orders the
        auxiliary (stop trigger) price is updated.

        Args:
            order_id: The string representation of the IB order ID.
            new_price: The new price to set.

        Returns:
            ``True`` if the modification was sent, ``False`` on error.
        """
        try:
            int_id = int(order_id)
        except ValueError:
            logger.error("Invalid order_id for modification: {}", order_id)
            return False

        for trade in self.ib.openTrades():
            if trade.order.orderId == int_id:
                order = trade.order
                if order.orderType == "LMT":
                    order.lmtPrice = new_price
                elif order.orderType in ("STP", "STP LMT"):
                    order.auxPrice = new_price
                else:
                    logger.warning(
                        "Cannot modify order type {} for order_id={}",
                        order.orderType,
                        order_id,
                    )
                    return False

                self.ib.placeOrder(trade.contract, order)
                logger.info(
                    "Modified order_id={} new_price={}", order_id, new_price
                )
                return True

        logger.warning("Order {} not found among open trades", order_id)
        return False

    def update_trailing_stop(self, trade_id: str, new_stop_price: float) -> bool:
        """Update the stop-loss leg of a bracket order to implement trailing behaviour.

        Looks up the bracket group by ``trade_id``, then modifies the stop
        order's auxiliary price.

        Args:
            trade_id: The application-level trade ID returned by
                ``submit_bracket_order``.
            new_stop_price: The new stop trigger price.

        Returns:
            ``True`` if the modification succeeded, ``False`` otherwise.
        """
        bracket = self._active_brackets.get(trade_id)
        if bracket is None:
            logger.warning("No active bracket found for trade_id={}", trade_id)
            return False

        sl_trade: Trade = bracket["sl_trade"]
        sl_order = sl_trade.order
        sl_order.auxPrice = new_stop_price

        try:
            self.ib.placeOrder(sl_trade.contract, sl_order)
            bracket["stop_loss"] = new_stop_price
            logger.info(
                "Trailing stop updated | trade_id={} new_stop={}",
                trade_id,
                new_stop_price,
            )

            # Persist to DB Trade record
            session = get_session()
            try:
                db_trade = (
                    session.query(DBTrade)
                    .filter(DBTrade.trade_id == trade_id)
                    .first()
                )
                if db_trade:
                    db_trade.trailing_stop_active = True
                    db_trade.trailing_stop_price = new_stop_price
                    db_trade.stop_loss_price = new_stop_price
                    db_trade.updated_at = datetime.utcnow()
                    session.commit()
            except Exception as db_exc:
                session.rollback()
                logger.error("DB error updating trailing stop: {}", db_exc)
            finally:
                session.close()

            return True
        except Exception as exc:
            logger.error("Failed to update trailing stop: {}", exc)
            return False

    def get_open_orders(self) -> List[Dict]:
        """Return all currently open orders from IBKR.

        Returns:
            A list of dicts, each containing ``order_id``, ``instrument``,
            ``action``, ``order_type``, ``quantity``, ``limit_price``,
            ``stop_price``, and ``status``.
        """
        result: List[Dict] = []
        for trade in self.ib.openTrades():
            order = trade.order
            result.append(
                {
                    "order_id": str(order.orderId),
                    "instrument": trade.contract.symbol,
                    "action": order.action,
                    "order_type": order.orderType,
                    "quantity": order.totalQuantity,
                    "limit_price": getattr(order, "lmtPrice", None),
                    "stop_price": getattr(order, "auxPrice", None),
                    "status": trade.orderStatus.status,
                }
            )
        return result

    def get_positions(self) -> List[Dict]:
        """Return current portfolio positions from IBKR.

        Returns:
            A list of dicts with ``instrument``, ``quantity``,
            ``avg_cost``, ``market_price``, ``market_value``,
            ``unrealized_pnl``, and ``realized_pnl``.
        """
        result: List[Dict] = []
        for pos in self.ib.positions():
            contract = pos.contract
            result.append(
                {
                    "instrument": contract.symbol,
                    "quantity": pos.position,
                    "avg_cost": pos.avgCost,
                    "market_price": None,  # filled below if available
                    "market_value": None,
                    "unrealized_pnl": None,
                    "realized_pnl": None,
                }
            )

        # Enrich from portfolio items which carry PnL
        for item in self.ib.portfolio():
            for entry in result:
                if entry["instrument"] == item.contract.symbol:
                    entry["market_price"] = item.marketPrice
                    entry["market_value"] = item.marketValue
                    entry["unrealized_pnl"] = item.unrealizedPNL
                    entry["realized_pnl"] = item.realizedPNL
                    break

        return result

    # --------------------------------------------------------------------- #
    #  Portfolio Info
    # --------------------------------------------------------------------- #

    def get_account_summary(self) -> Dict[str, float]:
        """Retrieve key account metrics from IBKR.

        Returns:
            A dictionary containing ``equity``, ``cash``, ``unrealized_pnl``,
            ``realized_pnl``, ``buying_power``, ``maintenance_margin``,
            ``gross_position_value``, and ``net_liquidation``.
        """
        tags = [
            "NetLiquidation",
            "TotalCashValue",
            "UnrealizedPnL",
            "RealizedPnL",
            "BuyingPower",
            "MaintMarginReq",
            "GrossPositionValue",
            "EquityWithLoanValue",
        ]
        values = self.ib.accountSummary()

        summary: Dict[str, float] = {
            "equity": 0.0,
            "cash": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "buying_power": 0.0,
            "maintenance_margin": 0.0,
            "gross_position_value": 0.0,
            "net_liquidation": 0.0,
        }

        tag_map = {
            "EquityWithLoanValue": "equity",
            "TotalCashValue": "cash",
            "UnrealizedPnL": "unrealized_pnl",
            "RealizedPnL": "realized_pnl",
            "BuyingPower": "buying_power",
            "MaintMarginReq": "maintenance_margin",
            "GrossPositionValue": "gross_position_value",
            "NetLiquidation": "net_liquidation",
        }

        for item in values:
            key = tag_map.get(item.tag)
            if key is not None:
                try:
                    summary[key] = float(item.value)
                except (ValueError, TypeError):
                    pass

        logger.debug("Account summary retrieved: {}", summary)
        return summary

    def get_portfolio_value(self) -> float:
        """Return the current net liquidation value of the account.

        Returns:
            Net liquidation value in account currency.
        """
        summary = self.get_account_summary()
        return summary["net_liquidation"]

    # --------------------------------------------------------------------- #
    #  Slippage Model (backtesting)
    # --------------------------------------------------------------------- #

    def apply_slippage(self, price: float, direction: str) -> float:
        """Apply a slippage model to a simulated fill price.

        Adjusts the price adversely by ``SLIPPAGE_PCT`` (default 0.02 %).
        Used in back-testing to approximate real-world execution costs.

        Args:
            price: The theoretical fill price.
            direction: ``"LONG"`` (slippage increases price) or ``"SHORT"``
                (slippage decreases price).

        Returns:
            The adjusted fill price.
        """
        if direction == "LONG":
            return round(price * (1 + self._slippage_pct), 2)
        return round(price * (1 - self._slippage_pct), 2)

    # --------------------------------------------------------------------- #
    #  Database Persistence
    # --------------------------------------------------------------------- #

    def _log_order_to_db(
        self,
        order_id: str,
        trade_id: str,
        instrument: str,
        order_type: OrderType,
        side: str,
        quantity: float,
        limit_price: Optional[float],
        stop_price: Optional[float],
    ) -> None:
        """Persist a new Order record to the database.

        Args:
            order_id: IBKR order ID (as string).
            trade_id: Application-level trade group ID.
            instrument: Ticker symbol.
            order_type: ``OrderType`` enum member.
            side: ``"BUY"`` or ``"SELL"``.
            quantity: Order quantity.
            limit_price: Limit price, if applicable.
            stop_price: Stop price, if applicable.
        """
        session = get_session()
        try:
            record = Order(
                order_id=order_id,
                trade_id=trade_id,
                instrument=instrument,
                order_type=order_type,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                submitted_at=datetime.utcnow(),
                status="SUBMITTED",
            )
            session.add(record)
            session.commit()
            logger.debug(
                "Order logged | order_id={} trade_id={} instrument={} side={} qty={}",
                order_id,
                trade_id,
                instrument,
                side,
                quantity,
            )
        except Exception as exc:
            session.rollback()
            logger.error("DB error logging order: {}", exc)
        finally:
            session.close()

    def _create_trade_record(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        """Create a Trade record in the database when a bracket order is submitted.

        Args:
            trade_id: Application-level trade group ID.
            instrument: Ticker symbol.
            direction: ``"LONG"`` or ``"SHORT"``.
            quantity: Number of contracts / shares.
            entry_price: Intended entry price.
            stop_loss: Stop-loss price.
            take_profit: Take-profit price.
        """
        session = get_session()
        try:
            td = TradeDirection.LONG if direction == "LONG" else TradeDirection.SHORT
            record = DBTrade(
                trade_id=trade_id,
                instrument=instrument,
                direction=td,
                status=TradeStatus.PENDING,
                entry_price=entry_price,
                entry_quantity=quantity,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(record)
            session.commit()
            logger.debug("Trade record created | trade_id={}", trade_id)
        except Exception as exc:
            session.rollback()
            logger.error("DB error creating trade record: {}", exc)
        finally:
            session.close()

    def _update_order_status_db(self, order_id: str, status: str) -> None:
        """Update the status field of an Order record in the database.

        Args:
            order_id: IBKR order ID (as string).
            status: New status string (e.g. ``"FILLED"``, ``"CANCELLED"``).
        """
        session = get_session()
        try:
            record = (
                session.query(Order)
                .filter(Order.order_id == order_id)
                .first()
            )
            if record:
                record.status = status
                if status == "CANCELLED":
                    record.cancelled_at = datetime.utcnow()
                session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("DB error updating order status: {}", exc)
        finally:
            session.close()

    def _update_order_fill_db(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: float,
        commission: float,
    ) -> None:
        """Record fill details on an Order record after execution.

        Args:
            order_id: IBKR order ID (as string).
            fill_price: Average fill price.
            fill_quantity: Filled quantity.
            commission: Commission charged for the fill.
        """
        session = get_session()
        try:
            record = (
                session.query(Order)
                .filter(Order.order_id == order_id)
                .first()
            )
            if record:
                record.fill_price = fill_price
                record.fill_quantity = fill_quantity
                record.commission = commission
                record.filled = True
                record.filled_at = datetime.utcnow()
                record.status = "FILLED"
                session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("DB error updating order fill: {}", exc)
        finally:
            session.close()

    # --------------------------------------------------------------------- #
    #  Event Handlers
    # --------------------------------------------------------------------- #

    def _on_order_status(self, trade: Trade) -> None:
        """Dispatch order-status updates to the appropriate handler.

        Connected to ``ib.orderStatusEvent``.  Routes to
        ``on_order_filled`` or ``on_order_cancelled`` based on the
        reported status.

        Args:
            trade: The ``ib_insync.Trade`` object whose status changed.
        """
        status = trade.orderStatus.status
        if status == "Filled":
            self.on_order_filled(trade)
        elif status in ("Cancelled", "ApiCancelled"):
            self.on_order_cancelled(trade)

    def on_order_filled(self, trade: Trade) -> None:
        """Handle a fully filled order.

        Logs the fill to the console and database, including timestamp,
        instrument, side, quantity, fill price, and commission.  If the
        filled order is the parent leg of a bracket, the corresponding
        Trade record is promoted to ``OPEN`` status.

        Args:
            trade: The filled ``ib_insync.Trade`` object.
        """
        order = trade.order
        fill_price = trade.orderStatus.avgFillPrice
        filled_qty = trade.orderStatus.filled
        commission = sum(fill.commissionReport.commission for fill in trade.fills if fill.commissionReport)

        logger.info(
            "ORDER FILLED | order_id={} instrument={} side={} qty={} "
            "fill_price={} commission={:.4f}",
            order.orderId,
            trade.contract.symbol,
            order.action,
            filled_qty,
            fill_price,
            commission,
        )

        # Persist fill to Order table
        self._update_order_fill_db(
            order_id=str(order.orderId),
            fill_price=fill_price,
            fill_quantity=filled_qty,
            commission=commission,
        )

        # If this is the parent order (entry), mark the Trade as OPEN
        for tid, bracket in self._active_brackets.items():
            parent_trade: Trade = bracket["parent_trade"]
            if parent_trade.order.orderId == order.orderId:
                session = get_session()
                try:
                    db_trade = (
                        session.query(DBTrade)
                        .filter(DBTrade.trade_id == tid)
                        .first()
                    )
                    if db_trade:
                        db_trade.status = TradeStatus.OPEN
                        db_trade.entry_time = datetime.utcnow()
                        db_trade.entry_price = fill_price
                        db_trade.entry_order_id = str(order.orderId)
                        db_trade.commission = (db_trade.commission or 0.0) + commission
                        db_trade.updated_at = datetime.utcnow()
                        session.commit()
                except Exception as exc:
                    session.rollback()
                    logger.error("DB error on fill update for trade {}: {}", tid, exc)
                finally:
                    session.close()
                break

            # Check if this is a TP or SL fill (exit leg)
            tp_trade: Trade = bracket["tp_trade"]
            sl_trade: Trade = bracket["sl_trade"]
            if order.orderId in (tp_trade.order.orderId, sl_trade.order.orderId):
                session = get_session()
                try:
                    db_trade = (
                        session.query(DBTrade)
                        .filter(DBTrade.trade_id == tid)
                        .first()
                    )
                    if db_trade and db_trade.entry_price is not None:
                        db_trade.status = TradeStatus.CLOSED
                        db_trade.exit_time = datetime.utcnow()
                        db_trade.exit_price = fill_price
                        db_trade.exit_order_id = str(order.orderId)
                        db_trade.commission = (db_trade.commission or 0.0) + commission

                        # Calculate realized PnL
                        if db_trade.direction == TradeDirection.LONG:
                            pnl = (fill_price - db_trade.entry_price) * db_trade.entry_quantity
                        else:
                            pnl = (db_trade.entry_price - fill_price) * db_trade.entry_quantity
                        db_trade.realized_pnl = pnl
                        db_trade.slippage = abs(fill_price * db_trade.entry_quantity) * self._slippage_pct
                        db_trade.updated_at = datetime.utcnow()
                        session.commit()
                except Exception as exc:
                    session.rollback()
                    logger.error("DB error on exit fill for trade {}: {}", tid, exc)
                finally:
                    session.close()

                # Remove bracket from active tracking
                del self._active_brackets[tid]
                break

    def on_order_cancelled(self, trade: Trade) -> None:
        """Handle a cancelled order.

        Logs the cancellation and updates the Order record in the database.

        Args:
            trade: The cancelled ``ib_insync.Trade`` object.
        """
        order = trade.order
        logger.warning(
            "ORDER CANCELLED | order_id={} instrument={} side={} qty={}",
            order.orderId,
            trade.contract.symbol,
            order.action,
            order.totalQuantity,
        )
        self._update_order_status_db(str(order.orderId), "CANCELLED")

        # If a bracket child is cancelled, update the Trade record
        for tid, bracket in list(self._active_brackets.items()):
            tp_trade: Trade = bracket["tp_trade"]
            sl_trade: Trade = bracket["sl_trade"]
            if order.orderId in (tp_trade.order.orderId, sl_trade.order.orderId):
                session = get_session()
                try:
                    db_trade = (
                        session.query(DBTrade)
                        .filter(DBTrade.trade_id == tid)
                        .first()
                    )
                    if db_trade:
                        db_trade.status = TradeStatus.CANCELLED
                        db_trade.updated_at = datetime.utcnow()
                        session.commit()
                except Exception as exc:
                    session.rollback()
                    logger.error("DB error on cancel for trade {}: {}", tid, exc)
                finally:
                    session.close()
                break

    def on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract = None) -> None:
        """Handle errors reported by the IBKR API.

        Non-critical informational codes (2100-series) are logged at debug
        level.  Connection-related errors trigger an automatic reconnection
        attempt.

        Args:
            reqId: The request ID associated with the error (-1 for global).
            errorCode: Numeric IBKR error code.
            errorString: Human-readable error description.
            contract: The contract related to the error, if any.
        """
        # Informational messages (not real errors)
        if 2100 <= errorCode <= 2110:
            logger.debug(
                "IBKR info | code={} msg={} reqId={}", errorCode, errorString, reqId
            )
            return

        logger.error(
            "IBKR ERROR | reqId={} code={} msg={} contract={}",
            reqId,
            errorCode,
            errorString,
            contract,
        )

        # Connection-loss error codes
        connection_error_codes = {1100, 1101, 1102, 504, 502}
        if errorCode in connection_error_codes:
            logger.warning("Connection error detected (code={}), scheduling reconnect", errorCode)
            self.connected = False
            asyncio.ensure_future(self.reconnect())

    def on_disconnected(self) -> None:
        """Handle an unexpected disconnection from IBKR.

        Sets the connected flag to ``False`` and initiates an automatic
        reconnection attempt.
        """
        logger.warning("Disconnected from IBKR -- initiating reconnect")
        self.connected = False
        asyncio.ensure_future(self.reconnect())
