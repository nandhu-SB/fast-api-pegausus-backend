from fastapi import FastAPI
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query  
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to handle NaN, None, inf
def clean_value(val):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return "N/A"
        return round(val, 2)
    return val

@app.get("/stock/{ticker}")
async def get_stock_data(
    ticker: str,
    history_period: str = Query("7d", description="History period, e.g., 7d, 1mo, 1y"),
    include_news: bool = Query(True, description="Include news articles"),

):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        news = stock.get_news()
        holders=stock.major_holders

        # 1-Year History
        # end_date = datetime.today()
        # start_date = end_date - timedelta(days=365)
        # history_data = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        history_data=stock.history(period=history_period)
        if history_data.empty:
            return {"error": "No price history available for the given ticker."}
        else:
            data = history_data.tail(1)

        # QoQ Financials
        qoq_financials = {}
        try:
            quarterly = stock.quarterly_financials.fillna(0)
            if not quarterly.empty:
                qoq_financials = quarterly.iloc[::-1].T.to_dict()
                # Choose specific features you want:
                selected_features = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
                filtered_quarterly={}
                for feature in selected_features:
                    if feature in qoq_financials:
                        filtered_quarterly[feature]=qoq_financials[feature]
                qoq_financials=filtered_quarterly
                

        except Exception as e:
            qoq_financials = {"error": "Quarterly financials not available"}

        # YoY Financials
        yoy_financials = {}
        try:
            yearly = stock.financials.fillna(0)
            if not yearly.empty:
                # Transpose & reverse order (most recent first)
                yearly_dict = yearly.iloc[::-1].T.to_dict()

                # Choose specific features you want:
                selected_features = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]

                # Filter the features
                filtered_financials = {}
                for feature in selected_features:
                    if feature in yearly_dict:
                        filtered_financials[feature] = yearly_dict[feature]

                yoy_financials = filtered_financials

        except Exception as e:
            yoy_financials = {"error": "Yearly financials not available"}


        # News Formatting
        formatted_news = []
        if include_news:
            if news:
                for article in news:
                    content = article.get('content', {})
                    title = content.get('title', '').strip()
                    link = content.get('clickThroughUrl', {}).get('url', '').strip()
                    if not title or not link:
                        continue
                    formatted_news.append({
                        "title": title,
                        "summary": content.get('summary', 'No Summary'),
                        "link": link,
                        "publisher": content.get('provider', {}).get('displayName', 'Unknown'),
                        "time": content.get('pubDate', 'N/A'),
                    })

        # Price History Formatting
        price_history = []
        if not history_data.empty:
            for index, row in history_data.iterrows():
                close_price = row["Close"]
                open_price=row["Open"]
                high_price=row["High"]
                low_price=row["Low"]
             
                # if pd.isna(close_price):
                #     continue

                price_history.append({
                    "date": index.strftime("%Y-%m-%d"),
                    "close": round(float(close_price), 2),
                    "open":round(float(open_price)),
                    "high":round(float(high_price)),
                    "low_price":round(float(low_price))
                    
                })

        # Final Response
        return {
            "symbol": info.get('longName',"N/A"),
            "sector":info.get('sector',"N/A"),
            "industry":info.get('industry',"N/A"),
            "business_summary": info.get("longBusinessSummary", "N/A"),
            "price": clean_value(data["Close"].iloc[-1]) if not data.empty else "N/A",
            "high": clean_value(data["High"].iloc[-1]) if not data.empty else "N/A",
            "low": clean_value(data["Low"].iloc[-1]) if not data.empty else "N/A",
            "open": clean_value(data["Open"].iloc[-1]) if not data.empty else "N/A",
            "volume": int(data["Volume"].iloc[-1]) if not data.empty else "N/A",
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
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
            "holders":holders
        }

    except Exception as e:
        return {"error": str(e)}
