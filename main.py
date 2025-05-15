from fastapi import FastAPI, Query
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import logging

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

# Helper function to clean values
def clean_value(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    return round(val, 2) if isinstance(val, float) else val

@app.get("/stock/{ticker}")
async def get_stock_data(
    ticker: str,
    history_period: str = Query("1y", description="History period, e.g., 7d, 1mo, 1y"),
    include_news: bool = Query(True, description="Include news articles"),
):
    try:
        stock = yf.Ticker(ticker)
        # if not stock.info:
        #     return {"error": f"Stock {ticker} not found."}

        # info = stock.info or {}
        try:
            info = stock.info
        except Exception as e:
            info={}

        news = stock.get_news() or []
        holders = stock.major_holders if not stock.major_holders.empty else "N/A"

        # Fetch history data
        history_data = stock.history(period=history_period)
        if history_data.empty:
            return {"error": "No price history available."}
        else:
            latest_data = history_data.iloc[-1]

        # QoQ Financials
        qoq_financials = {}
        try:
            quarterly = stock.quarterly_financials
            if not quarterly.empty:
                quarterly = quarterly.infer_objects(copy=False).fillna(0)
                qoq_financials = {k: clean_value(v) for k, v in quarterly.iloc[::-1].T.to_dict().items()}
        except Exception:
            qoq_financials = {"error": "Quarterly financials not available"}

        # YoY Financials
        yoy_financials = {}
        try:
            yearly = stock.financials
            if not yearly.empty:
                yearly = yearly.infer_objects(copy=False).fillna(0)
                yoy_financials = {k: clean_value(v) for k, v in yearly.iloc[::-1].T.to_dict().items()}
        except Exception:
            yoy_financials = {"error": "Yearly financials not available"}


        # QoQ balance_sheet
        qoq_balance_sheet = {}
        try:
            quarterly = stock.quarterly_balance_sheet
            if not quarterly.empty:
                quarterly = quarterly.infer_objects(copy=False).fillna(0)
                qoq_balance_sheet = {k: clean_value(v) for k, v in quarterly.iloc[::-1].T.to_dict().items()}
        except Exception:
            qoq_balance_sheet = {"error": "Quarterly balance_sheet not available"}

        # YoY balance_sheet
        yoy_balance_sheet = {}
        try:
            yearly = stock.balance_sheet
            if not yearly.empty:
                yearly = yearly.infer_objects(copy=False).fillna(0)
                yoy_balance_sheet = {k: clean_value(v) for k, v in yearly.iloc[::-1].T.to_dict().items()}
        except Exception:
            yoy_balance_sheet = {"error": "Yearly balance_sheet not available"}






        # QoQ cashflow
        # qoq_cashflow = {}
        # try:
        #     quarterly = stock.quarterly_cashflow
        #     if not quarterly.empty:
        #         quarterly = quarterly.infer_objects(copy=False).fillna(0)
        #         qoq_cashflow = {k: clean_value(v) for k, v in quarterly.iloc[::-1].T.to_dict().items()}
        # except Exception:
        #     qoq_cashflow = {"error": "Quarterly cashflow not available"}

        # YoY cashflow
        yoy_cashflow = {}
        try:
            yearly = stock.cashflow
            if not yearly.empty:
                yearly = yearly.infer_objects(copy=False).fillna(0)
                yoy_cashflow = {k: clean_value(v) for k, v in yearly.iloc[::-1].T.to_dict().items()}
        except Exception:
            yoy_cashflow = {"error": "Yearly cashflow not available"}





        # Sustainability Score
        sustainability_score = {}
        try:
            sustainability = stock.sustainability
            if not sustainability.empty:
                sustainability = sustainability.infer_objects(copy=False).fillna(0)
                sustainability_score = {
                    k: clean_value(v["esgScores"]) if isinstance(v, dict) and "esgScores" in v else clean_value(v)
                    for k, v in sustainability.iloc[::-1].T.to_dict().items()
                }
        except Exception:
            sustainability_score = {"error": "Sustainability Score is not available"}

        # Format news
        formatted_news = [
            {
                "title": a.get("content", {}).get("title", "No Title"),
                "summary": a.get("content", {}).get("summary", "No Summary"),
                "link": a.get("content", {}).get("clickThroughUrl", {}).get("url", ""),
                "publisher": a.get("content", {}).get("provider", {}).get("displayName", "Unknown"),
                "time": a.get("content", {}).get("pubDate", "N/A"),
            }
            for a in news if a.get("content")
        ]

        # Format price history
        
        price_history = [
            {
                "date": index.strftime("%Y-%m-%d"),
                "close": clean_value(row["Close"]),
                "open": clean_value(row["Open"]),
                "high": clean_value(row["High"]),
                "low": clean_value(row["Low"]),
            }
            for index, row in history_data.iterrows()
        ]

        # Process analyst recommendations safely
        recommendations = []
        try:
            rec_data = stock.recommendations
            # logger.info(f"Raw Recommendations Data for {ticker}: {rec_data}")

            if rec_data is not None and not rec_data.empty:
                recommendations = [
                    {
                        "period": row.Index.strftime("%Y-%m") if hasattr(row.Index, "strftime") else str(row.Index),
                        "strongBuy": int(row.strongBuy) if hasattr(row, "strongBuy") and not np.isnan(row.strongBuy) else 0,
                        "buy": int(row.buy) if hasattr(row, "buy") and not np.isnan(row.buy) else 0,
                        "hold": int(row.hold) if hasattr(row, "hold") and not np.isnan(row.hold) else 0,
                        "sell": int(row.sell) if hasattr(row, "sell") and not np.isnan(row.sell) else 0,
                        "strongSell": int(row.strongSell) if hasattr(row, "strongSell") and not np.isnan(row.strongSell) else 0,
                    }
                    for row in rec_data.tail(4).itertuples()
                ]
            else:
                logger.warning(f"No recommendations data available for {ticker}")
        except Exception as e:
            logger.error(f"Error processing recommendations for {ticker}: {str(e)}")
            recommendations = {"error": "Recommendations data not available"}

        return {
            "symbol": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "business_summary": info.get("longBusinessSummary", "N/A"),
            "price": clean_value(latest_data["Close"]),
            "high": clean_value(latest_data["High"]),
            "low": clean_value(latest_data["Low"]),
            "open": clean_value(latest_data["Open"]),
            "volume": int(latest_data["Volume"]) if not np.isnan(latest_data["Volume"]) else "N/A",
            "market_cap": clean_value(info.get("marketCap")),
            "pe_ratio": clean_value(info.get("trailingPE")),
            "dividend_yield": clean_value(info.get("dividendYield")),
            "previous_close": clean_value(info.get("previousClose")),
            "52_week_high": clean_value(info.get("fiftyTwoWeekHigh")),
            "52_week_low": clean_value(info.get("fiftyTwoWeekLow")),
            "average_volume": clean_value(info.get("averageVolume")),
            "currency": info.get("currency", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "news": formatted_news,
            "history": price_history,
            "quarterly_financials": qoq_financials,
            "yearly_financials": yoy_financials,
            "quarterly_balance_sheet": qoq_balance_sheet,
            "yearly_balance_sheet": yoy_balance_sheet,
            "yearly_cashflow": yoy_cashflow,
            "holders": holders,
            "sustainability_score": sustainability_score,
            "recommendations": recommendations,  # Fixed key name
        }
    
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
        return {"error": "An error occurred while fetching stock data."}
