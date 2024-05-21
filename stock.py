from polygon import RESTClient
import key

api_key = key.gcp_secret_polygon_stock_api_kay().payload.data.decode("utf-8")


def fetch_stock_data(ticker, start_date, end_date):
    client = RESTClient(api_key=api_key)
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day", sort='asc',
                              from_=start_date, to=end_date, limit=50000):
        aggs.append(a)
    return aggs


if __name__ == '__main__':
    result = fetch_stock_data('GOOG', '2024-05-01', '2024-05-01')
    for r in result:
        print(r)
    print(len(result))
