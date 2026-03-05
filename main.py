from fastapi import FastAPI, Query
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import logging
import traceback

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set pandas option to suppress future warnings
pd.set_option("future.no_silent_downcasting", True)


# ─── Helper functions ────────────────────────────────────────────────────────

def clean_value(val):
    """Clean a single value: convert NaN/Inf to None, round floats."""
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if isinstance(val, float):
        return round(val, 2)
    return val


def clean_dict(d):
    """Recursively clean all values in a dict."""
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    return clean_value(d)


def safe_dataframe_to_dict(df):
    """Safely convert a financials DataFrame to a cleaned dict."""
    if df is None or df.empty:
        return {}
    df = df.infer_objects(copy=False).fillna(0)
    raw = df.iloc[::-1].T.to_dict()
    return {k: clean_dict(v) for k, v in raw.items()}


def get_info_safe(stock):
    """Safely fetch stock.info."""
    try:
        return stock.info or {}
    except Exception as e:
        logger.warning(f"Failed to fetch info: {e}")
        return {}


def get_holders_safe(stock):
    """Safely fetch and serialize major_holders."""
    try:
        holders = stock.major_holders
        if holders is not None and not holders.empty:
            return holders.to_dict()
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch holders: {e}")
        return None


def get_financials_safe(stock, attr_name, label):
    """Safely fetch any financials attribute (quarterly_financials, balance_sheet, etc.)."""
    try:
        df = getattr(stock, attr_name, None)
        return safe_dataframe_to_dict(df)
    except Exception as e:
        logger.warning(f"Failed to fetch {label}: {e}")
        return {"error": f"{label} not available"}


def get_sustainability_safe(stock):
    """Safely fetch sustainability data."""
    try:
        sustainability = stock.sustainability
        if sustainability is not None and not sustainability.empty:
            sustainability = sustainability.infer_objects(copy=False).fillna(0)
            result = {}
            for k, v in sustainability.iloc[::-1].T.to_dict().items():
                if isinstance(v, dict) and "esgScores" in v:
                    result[k] = clean_value(v["esgScores"])
                else:
                    result[k] = clean_value(v)
            return result
        return {}
    except Exception as e:
        logger.warning(f"Failed to fetch sustainability: {e}")
        return {}


def get_news_safe(stock):
    """Safely fetch and format news."""
    try:
        news = stock.get_news() or []
        return [
            {
                "title": a.get("content", {}).get("title", "No Title"),
                "summary": a.get("content", {}).get("summary", "No Summary"),
                "link": a.get("content", {}).get("clickThroughUrl", {}).get("url", ""),
                "publisher": a.get("content", {}).get("provider", {}).get("displayName", "Unknown"),
                "time": a.get("content", {}).get("pubDate", "N/A"),
            }
            for a in news if a.get("content")
        ]
    except Exception as e:
        logger.warning(f"Failed to fetch news: {e}")
        return []


def get_recommendations_safe(stock, ticker):
    """Safely fetch and format analyst recommendations. Always returns a list."""
    try:
        rec_data = stock.recommendations
        if rec_data is None or rec_data.empty:
            logger.warning(f"No recommendations data available for {ticker}")
            return []

        recommendations = []
        for row in rec_data.tail(4).itertuples():
            recommendations.append({
                "period": row.Index.strftime("%Y-%m") if hasattr(row.Index, "strftime") else str(row.Index),
                "strongBuy": int(row.strongBuy) if hasattr(row, "strongBuy") and not np.isnan(row.strongBuy) else 0,
                "buy": int(row.buy) if hasattr(row, "buy") and not np.isnan(row.buy) else 0,
                "hold": int(row.hold) if hasattr(row, "hold") and not np.isnan(row.hold) else 0,
                "sell": int(row.sell) if hasattr(row, "sell") and not np.isnan(row.sell) else 0,
                "strongSell": int(row.strongSell) if hasattr(row, "strongSell") and not np.isnan(row.strongSell) else 0,
            })
        return recommendations
    except Exception as e:
        logger.error(f"Error processing recommendations for {ticker}: {e}")
        return []


def format_price_history(history_data):
    """Format price history DataFrame into a list of dicts."""
    return [
        {
            "date": index.strftime("%Y-%m-%d"),
            "close": clean_value(row["Close"]),
            "open": clean_value(row["Open"]),
            "high": clean_value(row["High"]),
            "low": clean_value(row["Low"]),
        }
        for index, row in history_data.iterrows()
    ]


def safe_latest(latest_data, key):
    """Safely extract a value from the latest price data row."""
    if latest_data is None:
        return None
    try:
        val = latest_data[key]
        if isinstance(val, float) and np.isnan(val):
            return None
        return clean_value(val)
    except Exception:
        return None


# ─── Main endpoint ───────────────────────────────────────────────────────────

@app.get("/stock/{ticker}")
async def get_stock_data(
    ticker: str,
    period: str = Query("1y", description="History period, e.g., 7d, 1mo, 1y"),
    include_news: bool = Query(True, description="Include news articles"),
):
    try:
        stock = yf.Ticker(ticker)

        # ── Info (non-fatal) ──
        info = get_info_safe(stock)

        # ── Price history (non-fatal — other sections still returned) ──
        latest_data = None
        price_history = []
        try:
            history_data = stock.history(period=period)
            if not history_data.empty:
                latest_data = history_data.iloc[-1]
                price_history = format_price_history(history_data)
            else:
                logger.warning(f"No price history returned for {ticker}")
        except Exception as e:
            logger.error(f"Failed to fetch history for {ticker}: {e}")

        # ── Financials (all non-fatal) ──
        qoq_financials = get_financials_safe(stock, "quarterly_financials", "Quarterly financials")
        yoy_financials = get_financials_safe(stock, "financials", "Yearly financials")
        qoq_balance_sheet = get_financials_safe(stock, "quarterly_balance_sheet", "Quarterly balance sheet")
        yoy_balance_sheet = get_financials_safe(stock, "balance_sheet", "Yearly balance sheet")
        yoy_cashflow = get_financials_safe(stock, "cashflow", "Yearly cashflow")

        # ── Other data (all non-fatal) ──
        holders = get_holders_safe(stock)
        sustainability_score = get_sustainability_safe(stock)
        recommendations = get_recommendations_safe(stock, ticker)
        news = get_news_safe(stock) if include_news else []

        # ── Volume ──
        volume = None
        if latest_data is not None:
            try:
                vol = latest_data["Volume"]
                volume = int(vol) if not np.isnan(vol) else None
            except Exception:
                volume = None

        return {
            "symbol": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "business_summary": info.get("longBusinessSummary", "N/A"),
            "price": safe_latest(latest_data, "Close"),
            "high": safe_latest(latest_data, "High"),
            "low": safe_latest(latest_data, "Low"),
            "open": safe_latest(latest_data, "Open"),
            "volume": volume,
            "market_cap": clean_value(info.get("marketCap")),
            "pe_ratio": clean_value(info.get("trailingPE")),
            "dividend_yield": clean_value(info.get("dividendYield")),
            "previous_close": clean_value(info.get("previousClose")),
            "52_week_high": clean_value(info.get("fiftyTwoWeekHigh")),
            "52_week_low": clean_value(info.get("fiftyTwoWeekLow")),
            "average_volume": clean_value(info.get("averageVolume")),
            "currency": info.get("currency", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "news": news,
            "history": price_history,
            "quarterly_financials": qoq_financials,
            "yearly_financials": yoy_financials,
            "quarterly_balance_sheet": qoq_balance_sheet,
            "yearly_balance_sheet": yoy_balance_sheet,
            "yearly_cashflow": yoy_cashflow,
            "holders": holders,
            "sustainability_score": sustainability_score,
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {traceback.format_exc()}")
        return {"error": f"Error for {ticker}: {str(e)}"}