"""Plotly Interactive Security Analytics & KPI Charts."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

class SecurityPlotter:
    """Generates interactive, dark-themed Plotly charts for security forensics."""

    # Futuristic cybersecurity color palette
    THEME_COLORS = ["#FF3366", "#00FFAA", "#FFAA00", "#00E5FF", "#9D00FF"]

    @classmethod
    def plot_hourly_activity(cls, df: pd.DataFrame) -> go.Figure:
        """Visualizes incident frequency aggregated across 24 hours."""
        if df.empty or "timestamp" not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No security incidents logged yet", showarrow=False, font=dict(size=16, color="#888"))
            fig.update_layout(template="plotly_dark", height=320)
            return fig

        df_copy = df.copy()
        df_copy["hour"] = pd.to_datetime(df_copy["timestamp"]).dt.hour
        hourly_counts = df_copy.groupby(["hour", "event_type"]).size().reset_index(name="count")

        fig = px.bar(
            hourly_counts,
            x="hour",
            y="count",
            color="event_type",
            title="Hourly Security Breach Distribution (24h)",
            labels={"hour": "Hour of Day (00:00 - 23:00)", "count": "Incident Count", "event_type": "Threat Type"},
            color_discrete_map={"INTRUSION": "#FF3366", "LOITERING": "#FFAA00", "MOTION": "#00FFAA"},
            template="plotly_dark",
            barmode="group"
        )
        fig.update_layout(
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
            height=340,
            margin=dict(l=30, r=30, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @classmethod
    def plot_zone_distribution(cls, df: pd.DataFrame) -> go.Figure:
        """Renders a donut chart displaying violation shares by zone."""
        if df.empty or "zone_name" not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No zone violation data available", showarrow=False, font=dict(size=16, color="#888"))
            fig.update_layout(template="plotly_dark", height=320)
            return fig

        zone_counts = df["zone_name"].value_counts().reset_index()
        zone_counts.columns = ["zone_name", "count"]

        fig = px.pie(
            zone_counts,
            names="zone_name",
            values="count",
            hole=0.55,
            title="Breach Distribution by Spatial Zone",
            color_discrete_sequence=cls.THEME_COLORS,
            template="plotly_dark"
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        return fig

    @classmethod
    def plot_loitering_duration_histogram(cls, df: pd.DataFrame) -> go.Figure:
        """Histogram illustrating dwell duration distribution for loitering events."""
        if df.empty or "duration" not in df.columns or (df["duration"] == 0).all():
            fig = go.Figure()
            fig.add_annotation(text="No loitering dwell recordings", showarrow=False, font=dict(size=16, color="#888"))
            fig.update_layout(template="plotly_dark", height=320)
            return fig

        loiter_df = df[df["event_type"] == "LOITERING"]
        if loiter_df.empty:
            loiter_df = df[df["duration"] > 0]

        fig = px.histogram(
            loiter_df,
            x="duration",
            nbins=15,
            title="Loitering Dwell Duration Spread (Seconds)",
            labels={"duration": "Dwell Time (s)"},
            color_discrete_sequence=["#FFAA00"],
            template="plotly_dark"
        )
        fig.update_layout(
            height=340,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        return fig

    @classmethod
    def plot_gauge(cls, value: float, title: str, min_val: float = 0, max_val: float = 100, color: str = "#00FFAA") -> go.Figure:
        """Renders a sleek gauge indicator for live telemetry."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 16, "color": "#E0E0E0"}},
            gauge={
                "axis": {"range": [min_val, max_val], "tickcolor": "#555"},
                "bar": {"color": color},
                "bgcolor": "#1a1a24",
                "borderwidth": 1,
                "bordercolor": "#333",
            }
        ))
        fig.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
