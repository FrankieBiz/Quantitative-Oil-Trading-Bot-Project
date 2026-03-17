# White Paper Research & Learning Plan

A step-by-step guide to researching, testing, and understanding every component of the Oil Quant Bot **before** writing the white paper. Each phase builds on the last.

---

## Phase 1: Understand the Oil Market (Week 1-2)

Before writing about oil trading, you need to deeply understand what makes oil markets unique.

### What to Study

| Topic | Why It Matters | Resources to Read |
|-------|---------------|-------------------|
| WTI vs Brent | Your bot trades both — you need to explain why | Search: "WTI Brent spread explained" on Investopedia |
| Contango vs Backwardation | Your bot uses this as a feature (contango_flag) | Search: "oil futures contango backwardation term structure" |
| The 3:2:1 Crack Spread | Your bot computes this — you need to explain the refining economics | Search: "3:2:1 crack spread formula crude oil" |
| OPEC and supply shocks | Your system's biggest risk factor | Read: Kilian (2009) "Not All Oil Price Shocks Are Alike" |
| EIA weekly report | Your bot has an eia_surprise feature | Read the actual EIA report at eia.gov/petroleum/supply/weekly |
| Oil market microstructure | Volume, liquidity, trading hours for CL futures | CME Group CL contract specifications |

### Action Items
- [ ] Read 5 recent EIA weekly reports and understand the inventory numbers
- [ ] Watch the WTI/Brent spread for 2 weeks, note how it moves
- [ ] Read Kilian (2009) paper — it's the foundational paper on oil shocks
- [ ] Read Hamilton (2009) on the 2007-08 oil shock causes
- [ ] Look up CL futures contract specs on CME Group website (tick size, margin, hours)

---

## Phase 2: Learn the ML Foundations (Week 2-4)

You need to be able to explain every model decision in the white paper.

### XGBoost

| What to Learn | How to Learn It |
|---------------|----------------|
| How gradient boosting works (not just "it's a model") | Read Chapter 10 of "Elements of Statistical Learning" (free PDF) or watch StatQuest XGBoost videos on YouTube |
| Why 3-class (long/flat/short) vs binary | Run your model with 2 classes and 3 classes, compare precision/recall |
| What volatility-scaled labels do | Open `ml/signal_model.py`, trace how labels are created. Plot the label distribution with and without scaling |
| Feature importance (gain vs cover vs weight) | Train your model, extract all 3 importance types, see which features matter |
| Overfitting detection | Compare train accuracy vs test accuracy per fold in walk-forward |

### Action Items
- [ ] Run the XGBoost model on historical data and export per-fold metrics
- [ ] Plot train vs test accuracy across all walk-forward folds — **if train >> test, you're overfit**
- [ ] Print feature importance rankings and check if oil-specific features actually contribute
- [ ] Try removing sentiment features and re-running — does performance drop?
- [ ] Try removing oil-specific features — does performance drop?
- [ ] Read the original XGBoost paper: Chen & Guestrin (2016)

### Key Question to Answer
> "If I remove the oil-specific features and sentiment, does the model degrade significantly? If not, those features aren't earning their keep and shouldn't be highlighted in the paper."

---

## Phase 3: Understand Sentiment Analysis (Week 3-4)

### FinBERT

| What to Learn | How |
|---------------|-----|
| How BERT works (attention mechanism basics) | Watch "BERT explained" by Jay Alammar (blog post or YouTube) |
| What FinBERT adds on top of BERT | Read the FinBERT paper: Araci (2019) |
| Whether FinBERT is good enough for oil | Search for "CrudeBERT" paper — compare their approach to yours |
| How your CSS formula works | Read `signals/sentiment.py` line by line, trace the math manually |

### Action Items
- [ ] Run FinBERT on 10 real oil news headlines and check if the sentiment scores make sense
- [ ] Test edge cases: "OPEC cuts production" (should be bullish), "US shale output surges" (should be bearish) — does FinBERT get these right?
- [ ] Compute CSS manually for a sample day using the formula from the code
- [ ] Research CrudeBERT and note in your paper that it's a potential improvement
- [ ] Test: Does adding sentiment features to XGBoost actually improve out-of-sample accuracy?

### Key Question to Answer
> "Does the sentiment signal add predictive value beyond technical indicators alone? What's the marginal improvement?"

---

## Phase 4: Study Regime Detection (Week 4-5)

### Hurst Exponent

| What to Learn | How |
|---------------|-----|
| What the Hurst exponent measures | Read the Wikipedia article + Mandelbrot & Wallis (1968) |
| R/S analysis calculation | Read `learning/adaptive.py`, trace the Hurst calculation step by step |
| Why median smoothing matters | Compute raw Hurst on 100 days of CL data, plot it. Then plot the smoothed version. See the difference |
| What hysteresis prevents | Count how many regime switches occur with and without hysteresis over 6 months |

### Action Items
- [ ] Calculate Hurst exponent on WTI daily closes for 2019-2024
- [ ] Plot raw Hurst vs median-smoothed Hurst — this becomes a figure in your paper
- [ ] Count regime switches with hysteresis vs without — this is a key result
- [ ] Compare Hurst-based regime detection against simple ADX-based detection
- [ ] Read about the Schmitt trigger (electrical engineering) — your hysteresis is analogous to this
- [ ] Test: Does the bot perform better with regime detection on vs off?

### Key Question to Answer
> "How many false regime switches does the hysteresis prevent? What's the performance impact?"

---

## Phase 5: Understand Risk Management (Week 5-6)

### Kelly Criterion

| What to Learn | How |
|---------------|-----|
| The Kelly formula derivation | Read Kelly (1956) original paper — it's short and readable |
| Why half-Kelly | Read Thorp (2006) chapter on Kelly in practice |
| What happens with full Kelly vs half Kelly vs quarter Kelly | Simulate 1000 trade sequences with each — plot the equity distributions |
| How estimation error affects Kelly | With only 50 trades, how stable is the win rate estimate? |

### Action Items
- [ ] Simulate Kelly sizing: generate 1000 random trade sequences (55% win rate, 1.5:1 payoff). Compare full Kelly, half Kelly, quarter Kelly, and fixed 2% sizing. Plot all 4 equity curves
- [ ] Calculate how many trades you need for a reliable Kelly estimate (hint: at least 100)
- [ ] Read the `risk/manager.py` file end-to-end, understand each layer
- [ ] Trace a sample trade through all 7 risk layers — document what gets checked at each step
- [ ] Study VaR: understand the Monte Carlo approach in the risk manager

### Key Question to Answer
> "Can I explain to an investor exactly what happens when a losing streak hits — what circuit breakers fire and in what order?"

---

## Phase 6: Master Backtesting (Week 6-7)

### Walk-Forward Validation

| What to Learn | How |
|---------------|-----|
| Why random cross-validation is wrong for time series | Read López de Prado "Advances in Financial Machine Learning" Chapter 7 |
| Anchored vs sliding walk-forward | Read `backtest/engine.py`, understand the fold generation logic |
| What equity curve stitching does | Trace `_stitch_equity_curves` — why is scaling needed? |
| Monte Carlo bootstrap methodology | Understand `monte_carlo_simulation` — what does ruin probability mean? |

### Action Items
- [ ] Run the full walk-forward backtest on CL data (2019-2024)
- [ ] Record these metrics for your paper: total return, Sharpe, max DD, win rate, profit factor, % profitable folds
- [ ] Run the same strategy with random signals (100 Monte Carlo runs) — this is your noise baseline
- [ ] Run buy-and-hold WTI as a comparison
- [ ] Run a simple EMA crossover as a comparison
- [ ] Run Monte Carlo simulation on your equity curve — record ruin probability and confidence intervals
- [ ] **Critical**: Compare in-sample vs out-of-sample Sharpe. If OOS Sharpe < 50% of IS Sharpe, your model is likely overfit

### Key Question to Answer
> "Does my system beat random signals AND buy-and-hold AND simple MA crossover on out-of-sample data with realistic transaction costs?"

---

## Phase 7: Run the Key Experiments (Week 7-9)

These are the experiments whose results will fill the placeholder tables in the white paper.

### Experiment 1: Full System Backtest
```
python -m oil_quant_bot.backtest.runner --start 2019-01-01 --end 2024-12-31
```
Record: Total return, Sharpe, max DD, win rate, profit factor, trade count

### Experiment 2: Ablation Study
Run the system with each component removed, one at a time:
- No sentiment features → compare performance
- No oil-specific features → compare performance
- No regime detection → compare performance
- No adaptive threshold → compare performance
- Fixed position sizing (2%) instead of Kelly → compare performance

This tells you which components actually matter.

### Experiment 3: Feature Importance
- Extract XGBoost feature importances (gain, cover, weight)
- Identify the top 10 and bottom 10 features
- Remove bottom 10 features and re-run — does performance improve? (less overfitting)

### Experiment 4: Regime Analysis
- Break trades by detected regime (TRENDING / MEAN_REVERTING / CHOPPY)
- Compute win rate, avg P&L, profit factor per regime
- Verify that TRENDING trades perform best and CHOPPY trades perform worst (validating the regime multipliers)

### Experiment 5: Monte Carlo
- Run 10,000 bootstrap simulations on the equity curve
- Record: ruin probability, 95% CI, 99% CI, median max drawdown

### Experiment 6: Paper Trading
- Run the bot in paper mode for at least 2-4 weeks
- Compare paper results to backtest predictions
- If paper Sharpe is within 50% of backtest Sharpe, the backtest is credible

---

## Phase 8: Academic Reading List (Throughout)

### Must-Read Papers (for the literature review)

| Paper | Year | Why Read It |
|-------|------|-------------|
| Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" | 2016 | You use XGBoost — must cite and understand |
| Araci, "FinBERT: Financial Sentiment Analysis" | 2019 | You use FinBERT — must cite |
| Kilian, "Not All Oil Price Shocks Are Alike" | 2009 | Foundational oil market paper |
| Hamilton, "Causes and Consequences of the Oil Shock of 2007-08" | 2009 | Oil market dynamics |
| Kelly, "A New Interpretation of Information Rate" | 1956 | Kelly criterion original — short paper |
| Thorp, "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market" | 2006 | Practical Kelly — half-Kelly justification |
| Hurst, "Long-term Storage Capacity of Reservoirs" | 1951 | Hurst exponent original |
| López de Prado, "Advances in Financial Machine Learning" | 2018 | Walk-forward validation, overfitting, feature importance — the bible |
| Harvey & Liu, "Backtesting" | 2015 | Why backtests lie — must cite in limitations |
| McLean & Pontiff, "Does Academic Research Destroy Return Predictability?" | 2016 | Why published strategies decay — context for limitations |
| Tetlock, "Giving Content to Investor Sentiment" | 2007 | Sentiment in markets — foundational |
| Baker & Wurgler, "Investor Sentiment in the Stock Market" | 2007 | Sentiment as systematic factor |
| Baumeister & Kilian, "Real-Time Forecasts of the Real Price of Oil" | 2012 | Oil forecasting methodology |

### Good-to-Read

| Paper/Book | Why |
|------------|-----|
| Shreve, "Stochastic Calculus for Finance II" | Deep understanding of risk modeling |
| Schulman et al., "Proximal Policy Optimization Algorithms" | Your RL module uses PPO |
| Mandelbrot & Wallis, "Noah, Joseph, and Operational Hydrology" | Hurst exponent theory |

---

## Phase 9: Write the Paper (Week 9-12)

Only start writing after you have:
- [ ] All experiment results in hand
- [ ] Read at least the Must-Read papers
- [ ] Run paper trading for 2+ weeks
- [ ] Answers to all "Key Questions" above

### Writing Order (easiest to hardest)
1. **Appendix** — Just copy hyperparameters from settings.py (already done in LaTeX template)
2. **System Architecture** — Describe what you built (you know this)
3. **Feature Engineering** — List and explain the 27 features
4. **Backtesting Methodology** — Describe how you validated
5. **Results** — Fill in the placeholder tables with real numbers
6. **Risk Management** — Explain each layer
7. **Regime Detection** — This is your main contribution — spend the most time here
8. **Literature Review** — Write after reading the papers
9. **Introduction** — Write after everything else (you'll know what to emphasize)
10. **Abstract** — Write last (it summarizes everything)

---

## Phase 10: Where to Publish

| Venue | Audience | Difficulty | Timeline |
|-------|----------|------------|----------|
| **SSRN** (ssrn.com) | Finance practitioners + academics | Easy (no peer review) | Upload anytime |
| **arXiv** (q-fin section) | Quantitative researchers | Easy (no peer review) | Upload anytime |
| **Medium / Towards Data Science** | General tech audience | Easy | Anytime |
| **Journal of Financial Data Science** | Academic | Hard (peer reviewed) | 6-12 months |
| **Quantitative Finance** (Taylor & Francis) | Academic | Hard (peer reviewed) | 6-12 months |
| **Journal of Commodity Markets** | Academic (oil-specific) | Hard (peer reviewed) | 6-12 months |

**Recommended path**: Publish on SSRN first (free, instant, gets a DOI). Then if the results are strong, submit a polished version to a journal.

---

## Checklist: Am I Ready to Write?

- [ ] I can explain how XGBoost works without reading notes
- [ ] I can derive the Kelly formula from scratch
- [ ] I understand why random cross-validation is wrong for time series
- [ ] I've read at least 8 of the 13 Must-Read papers
- [ ] I have real backtest results (not just the placeholder tables)
- [ ] I've run paper trading for 2+ weeks and compared to backtest
- [ ] I can explain the Hurst exponent to a non-technical person
- [ ] I know which features in my model actually matter (ablation study done)
- [ ] I can explain the difference between my system and a generic trading bot
- [ ] My out-of-sample Sharpe ratio is > 0.5 (otherwise the paper tells a weaker story)

---

## LaTeX Paper Template

The white paper template with all sections, formulas, diagrams, and placeholder tables is ready at:

```
docs/whitepaper/paper.tex       ← Full LaTeX paper (compile with pdflatex)
docs/whitepaper/references.bib  ← 23 academic citations in BibTeX format
```

To compile (requires LaTeX installed):
```bash
cd docs/whitepaper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The paper is structured with an executive summary (for non-technical readers) followed by full academic rigor (for researchers/journals). All results tables have `[TO BE FILLED]` markers where your experiment data goes.
