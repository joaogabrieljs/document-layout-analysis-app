from project import create_app

app = create_app()
app.config["DEBUG"] = True
app.run()
