import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask
from flask_cors import CORS
from flask import render_template_string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)
cors = CORS(app)

path = "olist_dataset/"

customers_db = pd.read_csv(path + "olist_customers_dataset.csv")
order_items_db = pd.read_csv(path + "olist_order_items_dataset.csv")
order_payments_db = pd.read_csv(path + "olist_order_payments_dataset.csv")
orders_db = pd.read_csv(path + "olist_orders_dataset.csv")
products_db = pd.read_csv(path + "olist_products_dataset.csv")
sellers_db = pd.read_csv(path + "olist_sellers_dataset.csv")
category_db = pd.read_csv(path + "product_category_name_translation.csv")
geolocation_db = pd.read_csv(path + "olist_geolocation_dataset.csv")
product_reviews_db = pd.read_csv(path + "olist_order_reviews_dataset.csv")

orders_db = orders_db.drop(
    columns=['order_approved_at', 'order_delivered_customer_date', 'order_delivered_carrier_date'])
products_db = products_db.drop(
    columns=["product_weight_g", "product_name_lenght", "product_description_lenght", "product_photos_qty",
             "product_length_cm", "product_height_cm", "product_width_cm", "product_length_cm"])
product_reviews_db = product_reviews_db.drop(columns=['review_comment_title', 'review_comment_message'])

products_db = products_db.merge(category_db, on='product_category_name')
products_db = products_db.rename(columns={'product_category_name_english': 'product_category_name',
                                          'product_category_name': 'product_category_name_old'})
products_db = products_db.drop(columns=['product_category_name_old'])

dataset = {
    'Customers': customers_db,
    'Order Items': order_items_db,
    'Payments': order_payments_db,
    'Orders': orders_db,
    'Products': products_db,
    'Sellers': sellers_db,
    'Reviews': product_reviews_db,
    'Geolocation': geolocation_db,
    'Category': category_db
}

df1 = order_items_db.merge(order_payments_db, on='order_id')
df2 = df1.merge(products_db, on='product_id')
df3 = df2.merge(sellers_db, on='seller_id')
df4 = df3.merge(product_reviews_db, on='order_id')
df5 = df4.merge(orders_db, on='order_id')
Olist = df5.merge(customers_db, on='customer_id')

# converting date columns to datetime
date_columns = ['shipping_limit_date', 'review_creation_date', 'review_answer_timestamp', 'order_purchase_timestamp',
                'order_estimated_delivery_date']
for col in date_columns:
    Olist[col] = pd.to_datetime(Olist[col], format='%Y-%m-%d %H:%M:%S')

# cleaning up name columns, and engineering new/essential columns
Olist['customer_city'] = Olist['customer_city'].str.title()
Olist['seller_city'] = Olist['seller_city'].str.title()
Olist['product_category_name'] = Olist['product_category_name'].str.title()
Olist['payment_type'] = Olist['payment_type'].str.replace('_', ' ').str.title()
Olist['product_category_name'] = Olist['product_category_name'].str.replace('_', ' ')
Olist['review_response_time'] = (Olist['review_answer_timestamp'] - Olist['review_creation_date']).dt.days
# Olist['delivery_against_estimated'] = (Olist['order_estimated_delivery_date'] - Olist['order_delivered_customer_date']).dt.days
# Olist['product_size_cm'] = Olist['product_length_cm'] * Olist['product_height_cm'] * Olist['product_width_cm']
Olist['order_purchase_year'] = Olist.order_purchase_timestamp.apply(lambda x: x.year)
Olist['order_purchase_month'] = Olist.order_purchase_timestamp.apply(lambda x: x.month)
Olist['order_purchase_dayofweek'] = Olist.order_purchase_timestamp.apply(lambda x: x.dayofweek)
Olist['order_purchase_hour'] = Olist.order_purchase_timestamp.apply(lambda x: x.hour)
Olist['order_purchase_day'] = Olist['order_purchase_dayofweek'].map(
    {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
Olist['order_purchase_mon'] = Olist.order_purchase_timestamp.apply(lambda x: x.month).map(
    {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov',
     12: 'Dec'})
#
# @app.route('/graph1', methods=['GET'])
# def graph1():
#     global Olist
#     purchase_count = Olist.groupby(['order_purchase_day', 'order_purchase_hour']).nunique()['order_id'].unstack()
#
#     # Create the figure and axes
#     fig, ax = plt.subplots(figsize=(20,8))
#
#     # Generate the heatmap on the axes
#     sns.heatmap(purchase_count.reindex(index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']),
#                 cmap="YlGnBu", annot=True, fmt="d", linewidths=0.2, ax=ax)
#
#     ax.set_xlabel('Purchases/Hour')
#     ax.set_ylabel('Day of Week')
#     return fig.to_html

@app.route('/graph1', methods=['GET'])
def graph1():
    global Olist
    purchase_count = Olist.groupby(['order_purchase_day', 'order_purchase_hour']).nunique()['order_id'].unstack()
    heat = go.Heatmap(
        z=purchase_count,
        x=purchase_count.columns.values,
        y=purchase_count.index.values,
        colorscale='YlGnBu'
    )
    fig = go.Figure(data=heat)
    fig.update_layout(
        title="Sales Each Month & Each DayofWeek",
        xaxis_title="Purchases/Hour",
        yaxis_title="Day of Week",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="#7f7f7f"
        )
    )

    # Tính tổng số lượng đơn đặt hàng
    total_orders = Olist['order_id'].nunique()

    # Tạo đoạn văn bản thông báo về tổng số lượng đơn đặt hàng
    total_orders_text = f"<div class='total_order' style='border: 1px solid #ddd; padding: 10px; margin-top: 20px; width: 200px;height:50px; background-color: #EEE8AA;border-radius: 5px;border-radius: 5px;'>Total number of orders: {total_orders}</div>"

    # Chuyển cả hai biểu đồ thành HTML
    graph_html = fig.to_html(full_html=False)


    # Sử dụng template để hiển thị biểu đồ và thông báo trên cùng một trang
    return render_template_string("""
    {{ total_orders_text|safe }}
    {{ graph_html|safe }}
    """, graph_html=graph_html, total_orders_text=total_orders_text)


@app.route("/graph2", methods=["GET"])
def graph2():
    global Olist
    # creating a purchase day feature
    df = Olist.copy()
    # creating an aggregation
    sales_per_purchase_month = Olist.groupby(['order_purchase_month', 'order_purchase_mon', 'order_purchase_day'],
                                             as_index=False).payment_value.sum()
    sales_per_purchase_month = sales_per_purchase_month.sort_values(by=['order_purchase_month'], ascending=True)

    df = sales_per_purchase_month
    fig = px.line(df, x="order_purchase_mon", y="payment_value", color='order_purchase_day',
                  title='Sales Each Month & Each DayofWeek')

    fig.update_layout(
        title="Sales Each Month & Each DayofWeek",
        xaxis_title="Months",
        yaxis_title="Sales(in $$)",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="#7f7f7f"
        )
    )
    return fig.to_html()


@app.route("/graph3", methods=["GET"])
def graph3():
    global Olist
    # creating a purchase day feature
    df = Olist.copy()
    # creating an aggregation
    sales_per_purchase_month = Olist.groupby(['order_purchase_month', 'order_purchase_mon', 'order_purchase_day'], as_index=False).payment_value.sum()
    sales_per_purchase_month = sales_per_purchase_month.sort_values(by=['order_purchase_month'], ascending=True)
    df = sales_per_purchase_month
    fig = px.line(df, x="order_purchase_mon", y="payment_value", color='order_purchase_day', title='Sales Each Month & Each DayofWeek')

    fig.update_layout(
        title="Sales Each Month & Each DayofWeek",
        xaxis_title="Months",
        yaxis_title="Sales(in $$)",
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="#7f7f7f"
        )
    )
    return fig.to_html()

@app.route("/graph4", methods=["GET"])
def graph4():
    # creating an aggregation
    avg_score_per_category = Olist.groupby('product_category_name', as_index=False).agg({'review_score': ['count', 'mean']})
    avg_score_per_category.columns = ['Product Category', 'Number of Reviews', 'Average Review Ratings']

    # filtering to show only categories with more than 50 reviews
    avg_score_per_category = avg_score_per_category[avg_score_per_category['Number of Reviews'] > 100]
    avg_score_per_category = avg_score_per_category.sort_values(by='Number of Reviews', ascending=False)
    avg_score_per_category
    avg_ratings = avg_score_per_category[:20]
    fig = px.bar(avg_ratings, x='Product Category', y='Number of Reviews',
             hover_data=['Average Review Ratings'], color='Average Review Ratings',
             height=500)
    return fig.to_html()

@app.route("/graph5", methods=["GET"])
def graph5():
    sales_per_category = Olist.groupby(['order_purchase_mon', 'product_category_name'], as_index=False).payment_value.sum()
    sales_per_category = sales_per_category.sort_values(by=['payment_value'], ascending=False)
    sales_per_category.columns = ['Purchase Month','Product Category', 'Sales Revenue']
    df = sales_per_category
    fig = px.bar(df, y='Sales Revenue', x='Product Category', text='Sales Revenue', hover_data=['Purchase Month'])
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(barmode='stack',uniformtext_minsize=8, uniformtext_mode='hide')
    return fig.to_html()

@app.route("/graph6", methods=["GET"])
def graph6():
    df = Olist[Olist.order_purchase_year == 2018]
    sales_per_category = df.groupby(['product_category_name'], as_index=False).payment_value.sum()
    sales_per_category = sales_per_category.sort_values(by=['payment_value'], ascending=False)
    sales_per_category.columns = ['Product Category', 'Sales Revenue']

    sales_per_category = sales_per_category[:20]
    labels = sales_per_category['Product Category']
    values = sales_per_category['Sales Revenue']

    # Use `hole` to create a donut-like pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    return fig.to_html()

@app.route("/review", methods=["GET"])
def review():
    olist_order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
    review_data = olist_order_reviews.loc[:, ['review_comment_message', 'review_score']]
    review_data = review_data.dropna(subset=['review_comment_message'])
    review_data = review_data.reset_index(drop=True)
    new_data_prepped = initial_prep_pipeline.transform(review_data)

    predictions = e2e_pipeline.predict(new_data_prepped['review_comment_message'])
    prediction_probabilities = e2e_pipeline.predict_proba(new_data_prepped['review_comment_message'])


    sentiment_counts = pd.Series(predictions).value_counts()
    probabilites_counts = prediction_probabilities

    positive_count = sentiment_counts.get(1, 0)  # Get count of 1s, default to 0 if not found
    negative_count = sentiment_counts.get(0, 0)  # Get count of 0s, default to 0 if not found

    print("Probability (Positive): ", positive_count)
    print("Probability (Negative): ", negative_count)
    # print(probabilites_counts)

    for i, prediction in enumerate(predictions[:10]):
        print(f"Review {i+1}:")
        print(f"Predicted Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
        print(f"Probability (Positive): {prediction_probabilities[i][1]:.2f}")
        print(f"Probability (Negative): {prediction_probabilities[i][0]:.2f}")
        print("-" * 20)

if __name__ == '__main__':
    app.run()
