import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_features(input_path, output_dir):
    """
    Loads the cleaned data and creates features and aggregations.
    """
    # Load cleaned data
    df = pd.read_csv(input_path, index_col='date', parse_dates=True)

    # Time Features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week

    # Cost Calculation
    logistics_cost = 10  # EUR per order
    visit_cost = 15      # EUR per visit
    df['order_normalized'] = np.where(df['number_of_orders'] > 0, 1, 0)
    df['cost'] = df['order_normalized'] * logistics_cost + df['promotor_visits'] * visit_cost
    df['profit'] = df['income'] - df['cost']

    # Reorganize columns
    column_order = [
        'client_id', 'city', 'channel', 'promotor_id', 'number_of_orders',
        'volume', 'income', 'median_ticket', 'promotor_visits', 'promotor_calls',
        'order_normalized', 'cost', 'profit', 'month', 'week', 'day_of_week'
    ]
    df = df[column_order]

    # Monthly Aggregation
    aggregation_rules = {
        'city': lambda x: x.mode()[0] if not x.mode().empty else None,
        'channel': lambda x: x.mode()[0] if not x.mode().empty else None,
        'promotor_id': lambda x: x.mode()[0] if not x.mode().empty else None,
        'order_normalized': 'sum',
        'volume': 'sum',
        'income': 'sum',
        'cost': 'sum',
        'profit': 'sum',
        'median_ticket': ['median', 'min', 'max', 'std'],
        'promotor_visits': 'median',
        'promotor_calls': 'median'
    }
    clients_monthly = df.groupby(['client_id', 'month']).agg(aggregation_rules)
    clients_monthly.columns = ['_'.join(col).strip() for col in clients_monthly.columns.values]
    clients_monthly.rename(columns={
        'city_<lambda>': 'city',
        'channel_<lambda>': 'channel',
        'promotor_id_<lambda>': 'promotor_id',
        'order_normalized_sum': 'total_orders',
        'volume_sum': 'total_volume',
        'income_sum': 'total_income',
        'cost_sum': 'total_cost',
        'profit_sum': 'total_profit',
        'median_ticket_median': 'median_ticket',
        'promotor_visits_median': 'median_promotor_visits',
        'promotor_calls_median': 'median_promotor_calls'
    }, inplace=True)

    # Efficiency Calculation
    efficiency_monthly = clients_monthly[['total_orders', 'median_promotor_visits']].copy()
    efficiency_monthly['efficiency'] = efficiency_monthly['total_orders'] / efficiency_monthly['median_promotor_visits']
    efficiency_monthly['efficiency'] = efficiency_monthly['efficiency'].replace([np.inf, -np.inf], np.nan)

    scaler = MinMaxScaler()
    finite_mask = efficiency_monthly['efficiency'].notna()
    efficiency_monthly.loc[finite_mask, 'efficiency_scaled'] = scaler.fit_transform(efficiency_monthly.loc[finite_mask, ['efficiency']])
    inf_value = efficiency_monthly['efficiency_scaled'].max() + 0.5
    efficiency_monthly['efficiency_scaled'] = efficiency_monthly['efficiency_scaled'].fillna(inf_value)
    efficiency_monthly.rename(columns={'total_orders': 'frequency'}, inplace=True)

    clients_monthly = clients_monthly.merge(efficiency_monthly[['efficiency', 'efficiency_scaled']], on=['client_id', 'month'], how='left')

    # Client-level Aggregation
    clients = clients_monthly.groupby('client_id').agg({
        'city': 'first',
        'channel': 'first',
        'promotor_id': 'first',
        'total_orders': 'sum',
        'total_volume': 'sum',
        'total_income': 'sum',
        'total_cost': 'sum',
        'total_profit': 'sum',
        'median_ticket': 'median',
        'median_promotor_visits': 'sum',
        'median_promotor_calls': 'sum',
        'efficiency': 'median',
        'efficiency_scaled': 'median'
    })
    clients.rename(columns={
        'median_promotor_visits': 'total_promotor_visits',
        'median_promotor_calls': 'total_promotor_calls'
    }, inplace=True)

    # Create processed directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save final DataFrames
    df.to_csv(os.path.join(output_dir, 'orders.csv'), index=True)
    clients.to_csv(os.path.join(output_dir, 'clients.csv'), index=True)
    clients_monthly.to_csv(os.path.join(output_dir, 'clients_monthly.csv'), index=True)
    print(f"Processed data saved to {output_dir}")