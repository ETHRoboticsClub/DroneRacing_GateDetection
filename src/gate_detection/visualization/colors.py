"""Visualisation colour palette for the 8 gate corner types."""

# Outer corners = warm colours, inner = cool — keeps the two rings visually
# distinguishable in QuAdGate debug overlays.
CORNER_COLORS: dict[str, str] = {
    "outer_TL": "#FF4444",
    "outer_TR": "#FF8C00",
    "outer_BR": "#FFD700",
    "outer_BL": "#ADFF2F",
    "inner_TL": "#00FFCC",
    "inner_TR": "#00BFFF",
    "inner_BR": "#6666FF",
    "inner_BL": "#CC44FF",
}
