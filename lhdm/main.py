import typer
import core.train_diffusion
from core.train import train as train_autoencoder
from core.visualize import visualize
from core.sandbox import test

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)
app.command()(train_autoencoder)
app.command()(core.train_diffusion.train_diffusion)
app.command()(visualize)
app.command()(test)

if __name__ == "__main__":
    app()
