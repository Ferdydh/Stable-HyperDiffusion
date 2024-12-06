import typer
import core.train
import core.train_diffusion
import core.visualize

app = typer.Typer(pretty_exceptions_show_locals=False)
app.command()(core.train.train)
app.command()(core.train_diffusion.train_diffusion)
app.command()(core.visualize.visualize)

if __name__ == "__main__":
    app()
