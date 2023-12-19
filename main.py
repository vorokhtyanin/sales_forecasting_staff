import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# Налаштування сторінки Streamlit
st.set_page_config(page_title="Звіт за 2022-2023", page_icon=":bar_chart:", layout="wide")

# Зчитування даних з Excel-файлу
file = 'zvit.xlsx'
df = pd.read_excel(
    dtype={"Акція на товар": 'float'},
    io=file,
    nrows=123,
)
# Виведення заголовка та даних на основній сторінці
st.title(":bar_chart: Звіт за 2022-2023")
st.markdown("##")
st.dataframe(df)
st.markdown("---")

# Налаштування параметрів фільтрації за допомогою бічної панелі
st.sidebar.title("Опції фільтрації")
with st.sidebar.expander("Фільтрація звітності"):
    category = st.sidebar.multiselect(
        "Оберіть категорію:",
        options=df["Категорія"].unique(),
        default=[]
    )
    season = st.sidebar.multiselect(
        "Оберіть сезон:",
        options=df["Сезон"].unique(),
        default=[]
    )
    date = st.sidebar.multiselect(
        "Оберіть рік:",
        options=df["Дата"].dt.year.unique(),
        format_func=lambda x: str(x),
    )
    price = st.sidebar.slider(
        "Цінова категорія (грн):",
        min_value=df["Ціна"].min(),
        max_value=df["Ціна"].max(),
        value=(df["Ціна"].min(), df["Ціна"].max()),
        format="%f грн"
    )

# Налаштування параметрів для прогнозування лінійною регресією
st.sidebar.title("Прогнозування")
independent_variable = st.sidebar.selectbox(
    "Оберіть змінну для прогнозування:",
    options=["Місяць"],
    index=None
)
dependent_variable = st.sidebar.selectbox(
    "Оберіть змінну для прогнозування:",
    options=["Продано товару", "Сума продажів", "Закуплено товару", "Націнка"],
    index=None
)
forecast_months = st.sidebar.number_input(
    "Кількість місяців для прогнозу",
    min_value=0,
    max_value=12,
    value=0,
    step=1
)
st.markdown("###")

# Фільтрація даних згідно обраних параметрів
if len(category) >= 1 or len(season) >= 1 or len(date) >= 1:

    filtered_df = df.copy()

    if len(category) >= 1:
        filtered_df = filtered_df[filtered_df["Категорія"].isin(category)]
    if len(season) >= 1:
        filtered_df = filtered_df[filtered_df["Сезон"].isin(season)]
    if len(date) >= 1:
        filtered_df = filtered_df[filtered_df["Дата"].dt.year.isin(date)]

    filtered_df = filtered_df.query(
        "Ціна >= @price[0] & Ціна <= @price[1]"
    )

    filtered_df["Дата"] = filtered_df["Дата"].dt.strftime("%d/%m/%Y")

    filtered_df["Акція на товар"] = filtered_df["Акція на товар"].apply(lambda x: f"{x:.2%}")

    filtered_df["Націнка"] = filtered_df["Націнка"].apply(lambda x: f"{x:.2%}")

    profit = filtered_df["Сума продажів"] - filtered_df["Загальна вартість"] - filtered_df["Загальні витрати"]

    # Виведення основних показників на головній сторінці
    total_price_product = int(filtered_df["Сума продажів"].sum())
    total_sales = int(filtered_df["Сума продажів"].sum())
    total_sales_product = int(filtered_df["Продано товару"].sum())
    total_spending = int(filtered_df["Загальні витрати"].sum())
    total_profit = int(profit.sum())

    first_column, second_column, third_column, fourth_column = st.columns(4)

    with first_column:
        st.subheader("Загальна вартість товару:")
        st.markdown(f"{total_price_product: ,} грн")
    with second_column:
        st.subheader("Загальні продажі:")
        st.markdown(f"{total_sales: ,} грн")
    with third_column:
        st.subheader("Кількість проданих одиниць:")
        st.markdown(f"{total_sales_product} шт")
    with fourth_column:
        st.subheader("Загальні витрати:")
        st.markdown(f"{total_spending: ,} грн")

    # Виведення прибутку
    st.markdown("###")
    st.subheader(":heavy_dollar_sign: Прибуток:")
    st.subheader(f"{total_profit: ,} грн")

    # Виведення даних
    st.markdown("###")
    st.dataframe(filtered_df)

    # Виведення графіків
    st.markdown("___")
    st.subheader(":page_facing_up: Статистика продажів")

    left_column, right_column = st.columns(2)
    with left_column:
        fig = px.line(filtered_df, x='Дата', y='Продано товару', color='Категорія')
        fig.update_layout(
            xaxis_title='Дата',
            yaxis_title='Кількість проданих одиниць',
        )
        st.plotly_chart(fig)
    with right_column:
        col = px.pie(filtered_df, values='Продано товару', names='Категорія')
        st.plotly_chart(col)

    st.markdown("___")

    # Прогнозування та виведення результатів
    st.subheader(":chart_with_upwards_trend: Прогнозування")
    st.markdown("###")

    left_column, right_column = st.columns(2)

    if independent_variable and dependent_variable and forecast_months > 0:

        # --- Лінійна регресія ---
        X = filtered_df[independent_variable].values.reshape(-1, 1)
        Y = filtered_df[dependent_variable].values

        model = LinearRegression()
        model.fit(X, Y)

        # Отримання коефіцієнтів
        a = model.coef_[0]
        b = model.intercept_

        # --- Прогноз ---
        start_month = 1
        end_month = filtered_df[independent_variable].max() + forecast_months
        forecast_X = np.arange(start_month, end_month).reshape(-1, 1)
        forecast_Y = model.predict(forecast_X)

        # --- Розрахунок залишку ---
        residuals = Y - model.predict(X)

        # Коефіцієнт а0
        a0 = sum(residuals)

        # Розрахунок значення прогнозу періодичної складової
        t = np.arange(1, len(X) + 1)
        n = len(X)

        # --- Розрахунок значення прогнозу періодичної складової ---
        b0 = np.sum(residuals * np.cos(2 * np.pi * (t - 1) / n)) / np.sum(np.cos(2 * np.pi * (t - 1) / n) ** 2)
        b1 = np.sum(residuals * np.sin(2 * np.pi * (t - 1) / n)) / np.sum(np.sin(2 * np.pi * (t - 1) / n) ** 2)

        periodic_component = b0 * np.cos(2 * np.pi * (t - 1) / n) - b1 * np.sin(2 * np.pi * (t - 1) / n)

        # --- Розрахунок загальної формули моделі продажу ---
        sales_model = a * forecast_X + b + periodic_component

        # --- Прогноз на наступні місяці ---
        forecast_result = pd.DataFrame({
            independent_variable: forecast_X.flatten(),
            dependent_variable: a * forecast_X.flatten() + b + b0 * np.cos(
                2 * np.pi * (forecast_X.flatten() - 1) / n) - b1 * np.sin(2 * np.pi * (forecast_X.flatten() - 1) / n)
        })

        # --- Розрахунок SSR (Sum of Squared Remainders) ---
        SSR = np.sum(residuals ** 2)

        # --- Розрахунок SST (Sum of Squared Total) ---
        SST = np.sum((Y - np.mean(Y)) ** 2)  # Загальна сума квадратів

        # --- Розрахунок кількості параметрів у моделі ---
        k = 4

        # --- Розрахунок ступенів вільності ---
        df_between = k - 1
        df_within = len(X) - k
        df_total = len(X) - 1

        # --- Розрахунок F-статистики ---
        F_statistic = (SST - SSR) / (SSR / df_within)

        # --- Виведення результатів ---
        alpha = 0.05  # рівень значущості
        p_value = 1 - stats.f.cdf(F_statistic, df_between, df_within)

        # --- Розрахунок коефіцієнта детермінації (R^2) ---
        R_squared = model.score(X, Y)

        # --- Виведення результату ---
        percentage_R_squared = R_squared * 100

        # Виведення результатів прогнозу та основних показників
        if dependent_variable != "Націнка":
            forecast_result[dependent_variable] = forecast_result[dependent_variable].astype(int)
        else:
            forecast_result[dependent_variable] = (forecast_result[dependent_variable] * 100).round().astype(
                int).astype(str) + ' %'

        # --- Дані прогнозування ---
        with left_column:
            # Перевірка, чи вибрано 2023 рік або 2022 рік
            if 2023 in date and len(date) == 1:
                forecast_display = forecast_result[forecast_result[independent_variable] >= 13]
            elif 2022 in date and len(date) == 1:
                forecast_display = forecast_result[forecast_result[independent_variable] <= 13]
            else:
                # Відображення всього прогнозу, якщо обрані інші роки або не обрано жодного року
                forecast_display = forecast_result

            st.write('Прогноз на наступні місяці:')
            st.dataframe(forecast_display)

            # Порівняння з рівнем значущості
            if p_value < alpha:
                st.write("Модель є статистично значущою. Вона є адекватною.")
            else:
                st.write("Модель не є статистично значущою. Вона може бути неадекватною.")
            with st.expander("Статистична оцінка"):
                st.write(f'Коефіцієнт конкордації складає: **{percentage_R_squared:.2f}%**')
                st.write(f'F-статистика: {F_statistic:.4f}')
                st.write(f'P-значення: {p_value:.18f}')

        # --- Графік лінійної регресії ---
        with right_column:
            plt.figure(figsize=(12, 8))

            # Побудова графіка для фактичних даних
            plt.plot(X, Y, color='blue', marker='o', linestyle='-', label='Фактичні значення')

            # Відзначення частини графіка, що відноситься до прогнозу
            plt.plot(forecast_display[independent_variable], forecast_display[dependent_variable], color='orange',
                     marker='o', linestyle='-', label='Прогнозовані значення')

            plt.xlabel(independent_variable)
            plt.ylabel(dependent_variable)
            plt.title('Прогноз на основі сезонної складової')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
    else:
        # Якщо поля не заповнені, не відображаємо інформацію про прогнозування
        with left_column:
            st.write("Будь ласка, виберіть змінні для прогнозування та кількість місяців для прогнозу.")
