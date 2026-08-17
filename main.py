"""Entry point: launches the BioID Vision GUI. Run with `python main.py`."""

from bioid_vision.gui import Application


def main():
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()
