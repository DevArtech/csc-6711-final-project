# Auxiliary Prediction Heads — A Simple Explanation

---

## Start Here: What a Normal Recommender Does

Imagine you're trying to predict what movie someone will watch next.
You give the model their watch history, and it spits out a guess.

```
Watch history:
  Titanic ★★★★★
  The Notebook ★★★★
  Pride & Prejudice ★★★★★
         │
         │
         ▼
  ┌─────────────────────────────────────────┐
  │                                         │
  │   MODEL                                 │
  │   (reads the history, thinks about it,  │
  │    builds an internal summary)          │
  │                                         │
  └─────────────────────────────────────────┘
         │
         │
         ▼
   "Next movie: Casablanca"
```

That internal summary the model builds is called a **hidden state**.
It's just a list of numbers that captures "what we know about this user."

---

## The Problem: The Summary Is Too Shallow

If the model is only ever asked "what's next?", it only learns to keep
information that helps answer that one question.

It might learn:
```
✅ "This user likes romance movies"
✅ "This user rates things highly"
```

But completely ignore:
```
❌ "This user's taste has been shifting lately"
❌ "This user seems to prefer older films"
❌ "Genre pattern: romance → drama → documentary"
```

The model didn't bother learning those things because they weren't
required to answer "what's next?" well enough to lower the loss.

---

## The Fix: Give It Extra Questions to Answer

What if, after building the internal summary, you forced the model to
also answer a few *extra* questions — questions that require it to notice
richer patterns in the history?

```
Watch history:
  Titanic ★★★★★
  The Notebook ★★★★
  Pride & Prejudice ★★★★★
         │
         ▼
  ┌─────────────────────────────────────────┐
  │   MODEL                                 │
  │   builds internal summary h             │
  └────────────────┬────────────────────────┘
                   │
                   │  h (the internal summary)
                   │
       ┌───────────┼───────────────┐
       │           │               │
       ▼           ▼               ▼
  "Next movie?"  "What genres   "What star
                  does this       rating will
                  user like?"     they give?"

   MAIN TASK      EXTRA #1        EXTRA #2
```

The extra questions are the **auxiliary heads**.

---

## Why Does This Help?

To answer "what genres does this user like?", the internal summary
**must** encode genre information — there is no other way.

So the model is *forced* to build a richer summary — one that captures
genre patterns, rating tendencies, and more.

Then when the main question ("what's next?") is answered from that
same richer summary, it gets *better answers*, even though the genre
head is just a side task.

```
WITHOUT auxiliary heads:        WITH auxiliary heads:
─────────────────────────       ──────────────────────────
Summary h encodes:              Summary h encodes:
  • rough movie similarity        • rough movie similarity
                                  • genre preferences   ← bonus
                                  • rating level        ← bonus
                                  • taste trends        ← bonus

Recommendation quality:         Recommendation quality:
  okay                            better
```

The extra heads are thrown away after training.
They are scaffolding — they shape the building, then get removed.

---

## In This Project: What Already Exists

The sequential model (`sequential_model.py`) already has this structure:

```
User's watch history
  (items + ratings, in time order)
         │
         ▼
  ┌─────────────────┐
  │   GRU           │  ← reads history step by step,
  │   (backbone)    │    builds up a hidden state h
  └────────┬────────┘
           │
           │   h  (128 numbers summarizing the user)
           │
    ┌──────┼──────────────┐
    │      │              │
    ▼      ▼              ▼
"What   "What genres  "What star
 movie   does this     rating will
 next?"  user like?"   they give?"

  MAIN     AUX 1          AUX 2
```

And in the training loop (`train_sequential.py` line 106):

```python
total loss = next-item loss
           + 0.1 × genre loss
           + 0.1 × rating loss
```

One backward pass. All three losses push gradients back into the GRU.
The GRU gets nudged to keep genre and rating signals alive in h.

---

## What the Bayesian Model Does Instead

The Bayesian model (`bayesian_mf.py`) has no neural network heads.
It works differently — but achieves a similar goal.

Instead of a frozen hidden state h, it keeps a **probability cloud** per user:

```
After seeing: Titanic ★★★★★

  User belief before:                User belief after:

  ┌──────────────────────┐          ┌──────────────────────┐
  │  ?    ?    ?    ?    │          │  Romance ████  HIGH   │
  │  ?    ?    ?    ?    │  ──────► │  Action  ██    low    │
  │  ?    ?    ?    ?    │  update  │  Horror  █     low    │
  │  (wide uncertainty)  │          │  (narrower, shifted)  │
  └──────────────────────┘          └──────────────────────┘
```

Every new rating triggers a Bayes update. The model never freezes.
The forgetting factor (λ ≈ 0.95) means old ratings slowly fade —
if your taste changes, the belief catches up.

---

## Side-by-Side: The Three Models

```
┌────────────────────────────────────────────────────────────┐
│ STATIC MF                                                  │
│                                                            │
│  Trains once. Frozen forever.                              │
│  No auxiliary heads.                                       │
│  Belief = one fixed number per user, never changes.        │
│                                                            │
│  ● ─────────────────────────────────────────── time ►     │
│  trained                  (still the same dot)             │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ SEQUENTIAL (GRU)                                           │
│                                                            │
│  Trains once. Frozen after.                                │
│  Has auxiliary heads (genre + rating) during training.     │
│  At inference: reads full history, gives context-aware     │
│  predictions — but the model weights themselves are fixed. │
│                                                            │
│  ████ ─────────────────────────────────────── time ►      │
│  trained, rich hidden state, but frozen weights            │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ BAYESIAN MF  ★                                             │
│                                                            │
│  Trains once (item factors). Then updates forever.         │
│  No neural heads — Bayes rule IS the update mechanism.     │
│  Belief per user = probability distribution, not a dot.    │
│                                                            │
│  ████ ───●───●───●───●───●───●───●───● time ►             │
│  trained  each dot = one Bayes update after a rating       │
└────────────────────────────────────────────────────────────┘
```

---

## The Drift Scenario: Why Any of This Matters

```
A user switches from Horror to Documentaries halfway through the year:

time ──────────────────────┬──────────────────────►
         Horror phase      │    Documentary phase
                           │
                     taste changes here


STATIC MF:   [trained on Horror] ──────────────────────
             frozen. Still recommends Horror.         ✗

SEQUENTIAL:  [trained]  reads recent Docs in history ──
             sees the shift, does better.             ~

BAYESIAN:    [trained]  ●  ●  ●  [updating] ●  ●  ●  ──
             belief slowly shifts to Docs.            ✓
```

This is the whole experiment. The drift users subset in the results
(`drift_subset_summary.json`) is where the Bayesian model should win.

---

## One Addition That Would Make It Even Better

The sequential model does not know *when* a user's taste changed.
A **drift detection head** would force the GRU to notice:

```
           h  (GRU hidden state)
           │
    ┌──────┼──────────────┬──────────────┐
    │      │              │              │
    ▼      ▼              ▼              ▼
"Next   "Genres"      "Rating"    "Has this user's
 movie?"  bucket"      level"      taste SHIFTED
                                   recently?"

                                   output: 0.0–1.0
                                   DRIFT HEAD  ← new
```

Training target: already known from `data/drift_simulator.py`.
We know exactly which users drifted and at what point in time.

The GRU would be forced to encode "is this user in transition?" in h —
which is the exact signal the whole project is designed to measure.
