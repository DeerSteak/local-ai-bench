"""Engine Management screen composition."""

from scripts.app.engine_management import build_engine_management_tab


def build_engine_screen(notebook, *, ttk, **management_options):
    frame = ttk.Frame(notebook, padding=18)
    notebook.add(frame, text="Engine Management")
    controller = build_engine_management_tab(parent=frame, ttk=ttk, **management_options)
    return frame, controller
