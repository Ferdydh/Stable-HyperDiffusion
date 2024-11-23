import typer
import matplotlib

cmd = typer.Typer(pretty_exceptions_show_locals=False)
matplotlib.use("TkAgg")
