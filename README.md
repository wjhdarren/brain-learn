# brain-learn

A genetic programming framework for the [WorldQuant BRAIN platform](https://platform.worldquantbrain.com/), inspired by [gplearn](https://github.com/trevorstephens/gplearn).

## Setup

1.  **Environment:** Configure your Python environment using [`uv`](https://docs.astral.sh/uv/):
    ```bash
    uv sync
    ```
    This command creates a python virtual environment in `.venv` and installs the dependencies listed in `pyproject.toml`.

2.  **Configuration:**
    *   Create a `.env` file in the project root directory. The file should contain necessary credentials, particularly `USERNAME` and `PASSWORD`.
    *   Modify `main.py` to set your desired parameters for the genetic programming run (e.g., population size, generations, simulation settings).

3.  **Initial Population (Optional):** You can provide an initial population of expressions (signals) to kickstart the evolution process. Refer to the implementation in `main.py` or `src/genetic.py` for how to load or specify this initial population.

## Running the Framework

Execute the main script using `uv`:

```bash
uv run main.py
```

or activate the environment first then run with the corresponding intepreter

```bash
source .venv/bin/activate
python3 main.py
```

This will start the genetic programming process based on the configurations in `main.py`.

## Customization

You can customize the building blocks of the genetic programming process:

*   **Operators and Terminals:** Add or modify operators (functions like `add`, `ts_rank`) and terminals (input features like `close`, `volume`) in `src/function.py`.
*   **Weights:** Adjust the `weight` parameter for `Operator` and `Terminal` instances in `src/function.py` to influence their probability of being selected during the evolutionary process. Higher weights mean higher probability.

## Disclaimer

Notice: This codebase is experimental and intended solely for personal use. It is provided 'AS IS', without representation or warranty of any kind. Liability for any use or reliance upon this software is expressly disclaimed.




