import asyncio
import ipywidgets as widgets
from IPython.display import display

class HumanWidgetStrategy:
    """
    Replaces human_strategy_console.
    Call `get_action(board, mask, player)` which returns a coroutine that
    resolves only after the player clicks a column button.
    """
 
    def __init__(self):
        self._future: asyncio.Future | None = None
        self.button_row = widgets.HBox([])
 
    def _make_buttons(self, mask):
        buttons = []
        for col, allowed in enumerate(mask):
            btn = widgets.Button(
                description=str(col),
                disabled=(not allowed),
                button_style="primary" if allowed else "",
                layout=widgets.Layout(width="48px", height="48px"),
            )
            if allowed:
                btn.on_click(self._on_click(col))
            buttons.append(btn)
        self.button_row.children = buttons
 
    def _on_click(self, col):
        def handler(b):
            if self._future and not self._future.done():
                self._future.get_loop().call_soon_threadsafe(
                    self._future.set_result, col
                )
        return handler
 
    async def get_action(self, board, mask, player):
        loop = asyncio.get_event_loop()
        self._future = loop.create_future()
        self._make_buttons(mask)
        display(self.button_row)          # show column buttons
        col = await self._future          # wait here until a button is clicked
        self.button_row.children = []     # hide buttons after choice
        return col
    
    