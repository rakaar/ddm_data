---
trigger: always_on
---

<animal_sound_localization activation="always">

## Dataset location
All experimental CSVs live in  
`fit_animal_by_animal/batch_csvs/`  
Each file: `batch_{batch_name}_valid_and_aborts.csv`

## Trial types
* **Valid** – the animal responded *after* sound onset (row has `success ∈ {1, -1}`).
* **Abort** – the animal broke fixation and responded *before* sound onset (`abort_event == 3`).

## Column glossary
| column          | meaning                                                   | dtype |
|-----------------|-----------------------------------------------------------|-------|
| `batch_name`    | name of the batch                                         | str   |
| `animal`        | subject ID                                                | str   |
| `ABL`           | average sound level of both speakers                      | float |
| `ILD`           | right – left level (dB)                                   | float |
| `choice`        | 1 = right, −1 = left                                      | int   |
| `success`       | 1 = correct (reward), −1 = error (no reward)              | int   |
| `RTwrtStim`     | reaction time relative to stimulus onset (s)              | float |
| `intended_fix`  | stimulus-onset time relative to fixation start (s)        | float |
| `TotalFixTime`  | total fixation time (s) (≙ RT relative to fixation)       | float |

## Curve definitions

- **Psychometric curve**  
  Plots **probability of choosing right (`choice == 1`)** vs **ILD**, usually **for each ABL**.  
  Typically shown as a **scatter plot**. And with sigmoid fit passing through the points.

- **Chronometric curve**  
  Plots **mean `RTwrtStim`** vs **absolute ILD** for each ABL.  
  If the column `abs_ILD` is missing, create it using `abs(ILD)`.  
  Plot with **error bars showing standard deviation**, and **no caps** on the error bars. 

- **Tachometric curve**  
  Plots **accuracy (i.e., probability of `success == 1`)** as a function of RT.  
  Procedure:  
  1. Divide `RTwrtStim` into time bins (default = 20 ms unless user specifies otherwise).  
  2. In each bin, compute the **fraction of trials with `success == 1`** (i.e., correct responses).  
  3. Plot accuracy vs **bin centers**.

</animal_sound_localization>
