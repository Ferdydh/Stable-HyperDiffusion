import typer
import core.train
import core.visualize
import torch
import random
import numpy as np

app = typer.Typer(pretty_exceptions_show_locals=False)
app.command()(core.train.train)
app.command()(core.visualize.visualize)

if __name__ == "__main__":
    app()
