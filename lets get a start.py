from ast import main

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Sayfa yapılandırması
st.set_page_config(page_title="Start to Start-Up", page_icon=":rocket:", layout="wide")

# Başlık ve açıklama
st.title('Start to Start-Up :rocket:')
st.write('Hoş geldiniz! Bu site, veri analizleri yapmanıza ve start-up fikrinizi geliştirmenize yardımcı olacak araçları sunmaktadır.')

# Menü
menu = st.sidebar.selectbox('Seçiminizi Yapın', ['Ana Sayfa', 'Analizler', 'Veri Yükleme', 'Start-Up Fikri'])

if menu == 'Ana Sayfa':
    st.image('start up.jpg', use_column_width=True)
    st.header('Amacımız')
    st.write(
        """
        Amacımız, veri analizleri ve start-up fikirlerinizi geliştirmek için çeşitli araçlar ve kaynaklar sağlamaktır. 
        Sitenin içindeki bölümlerle, veri yükleyebilir, analizler gerçekleştirebilir ve start-up fikrinizi şekillendirebilirsiniz.
        """
    )

elif menu == 'Analizler':
    st.header('Veri Analizleri')

    sub_menu = st.selectbox('Analiz Seçin', ['İstihdam Analizi',
                                             'Endeks Analizi','Gelecek Yıllara Dair İthalat-İhracat Tahmini',
                                             'Ekonomik Faaliyetlere Göre Temel Göstergeler 2015-2022',
                                             'Startup Yatırım Analizi', 'Bilanço Analizi'])

    if sub_menu == 'İstihdam Analizi':
        st.write("## İstihdam Analizi")

        # İstihdam analizi için veri oluşturma
        data = {
            'Year': np.arange(1991, 2023),
            'Employment_in_services': [
                31.949, 33.552, 35.792, 35.260, 35.426, 35.275, 36.124, 36.547, 38.061, 40.125,
                39.747, 42.127, 42.979, 45.963, 47.954, 49.171, 49.775, 49.521, 51.750, 50.073,
                49.380, 50.417, 50.686, 51.060, 52.370, 53.724, 54.079, 54.905, 56.572, 56.205, 55.310, 55.609
            ],
            'Employment_in_manufacturing': [
                20.290, 21.644, 21.713, 20.690, 20.466, 21.038, 22.200, 21.949, 21.761, 22.950,
                22.662, 22.913, 23.412, 24.936, 26.361, 26.771, 26.748, 26.810, 25.303, 26.223,
                26.462, 26.028, 26.391, 27.856, 27.225, 26.775, 26.538, 26.665, 25.319, 26.241, 27.506, 27.731
            ],
            'Employment_in_IT': [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0.00875400100266091, 0.0094290007513148, 0.00911597132879985,
                0.00915250629678243, 0.00803729863261543, 0.00833244311505181,
                0.00898817737664566, 0.00857758021947493, 0.00916983708906448
            ]
        }
        df = pd.DataFrame(data)

        # Tarih sütunu ekle
        df['Date'] = pd.to_datetime(df['Year'], format='%Y')
        df.set_index('Date', inplace=True)

        # ARIMA ve Prophet için tahmin fonksiyonları
        def arima_forecast(df, column, forecast_steps):
            model = ARIMA(df[column], order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.get_forecast(steps=forecast_steps)
            forecast_mean = forecast.predicted_mean
            forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
            conf_int = forecast.conf_int()
            return forecast_mean, forecast_index, conf_int

        def prophet_forecast(df, column, forecast_steps):
            prophet_df = df.reset_index().rename(columns={'Date': 'ds', column: 'y'})
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=forecast_steps, freq='ME')
            forecast = model.predict(future)
            return forecast

        # Tahminler
        forecast_steps = 120  # 10 yıl (120 ay) tahmin et

        # ARIMA tahminleri
        arima_forecast_services, arima_index_services, arima_conf_services = arima_forecast(df, 'Employment_in_services', forecast_steps)
        arima_forecast_manufacturing, arima_index_manufacturing, arima_conf_manufacturing = arima_forecast(df, 'Employment_in_manufacturing', forecast_steps)
        arima_forecast_IT, arima_index_IT, arima_conf_IT = arima_forecast(df, 'Employment_in_IT', forecast_steps)

        # Prophet tahminleri
        prophet_forecast_services = prophet_forecast(df, 'Employment_in_services', forecast_steps)
        prophet_forecast_manufacturing = prophet_forecast(df, 'Employment_in_manufacturing', forecast_steps)
        prophet_forecast_IT = prophet_forecast(df, 'Employment_in_IT', forecast_steps)

        # Zaman Serisi Grafiği
        st.write("### Zaman Serisi Grafiği")
        fig_ts, ax_ts = plt.subplots(figsize=(14, 8))
        ax_ts.plot(df.index, df['Employment_in_services'], label='Hizmetler')
        ax_ts.plot(df.index, df['Employment_in_manufacturing'], label='İmalat')
        ax_ts.plot(df.index, df['Employment_in_IT'], label='Bilişim')
        ax_ts.set_xlabel('Tarih')
        ax_ts.set_ylabel('İndeks')
        ax_ts.set_title('Zaman Serisi Grafiği')
        ax_ts.legend()
        st.pyplot(fig_ts)

        # Genel Trendler Grafiği
        st.write("### Genel Trendler")
        fig_trends, ax_trends = plt.subplots(figsize=(14, 8))
        ax_trends.plot(df.index, df['Employment_in_services'], label='Hizmetler')
        ax_trends.plot(df.index, df['Employment_in_manufacturing'], label='İmalat')
        ax_trends.plot(df.index, df['Employment_in_IT'], label='Bilişim')
        ax_trends.set_xlabel('Tarih')
        ax_trends.set_ylabel('İndeks')
        ax_trends.set_title('Genel Trendler')
        ax_trends.legend()
        st.pyplot(fig_trends)

        # ARIMA Modeli ile Gelecek Tahminleri
        st.write("### ARIMA Modeli ile Gelecek Tahminleri")
        fig_arima, ax_arima = plt.subplots(figsize=(14, 8))
        ax_arima.plot(df.index, df['Employment_in_services'], label='Hizmetler (Gerçek)')
        ax_arima.plot(df.index, df['Employment_in_manufacturing'], label='İmalat (Gerçek)')
        ax_arima.plot(df.index, df['Employment_in_IT'], label='Bilişim (Gerçek)')
        ax_arima.plot(arima_index_services, arima_forecast_services, label='ARIMA Hizmetler Tahmini', color='red')
        ax_arima.plot(arima_index_manufacturing, arima_forecast_manufacturing, label='ARIMA İmalat Tahmini', color='blue')
        ax_arima.plot(arima_index_IT, arima_forecast_IT, label='ARIMA Bilişim Tahmini', color='green')
        ax_arima.fill_between(arima_index_services, arima_conf_services.iloc[:, 0], arima_conf_services.iloc[:, 1], color='red', alpha=0.1)
        ax_arima.fill_between(arima_index_manufacturing, arima_conf_manufacturing.iloc[:, 0], arima_conf_manufacturing.iloc[:, 1], color='blue', alpha=0.1)
        ax_arima.fill_between(arima_index_IT, arima_conf_IT.iloc[:, 0], arima_conf_IT.iloc[:, 1], color='green', alpha=0.1)
        ax_arima.set_xlabel('Tarih')
        ax_arima.set_ylabel('İndeks')
        ax_arima.set_title('ARIMA Modeli ile Gelecek Tahminleri')
        ax_arima.legend()
        st.pyplot(fig_arima)

        # Prophet Modeli ile Gelecek Tahminleri
        st.write("### Prophet Modeli ile Gelecek Tahminleri")
        fig_prophet, ax_prophet = plt.subplots(figsize=(14, 8))
        ax_prophet.plot(df.index, df['Employment_in_services'], label='Hizmetler (Gerçek)')
        ax_prophet.plot(df.index, df['Employment_in_manufacturing'], label='İmalat (Gerçek)')
        ax_prophet.plot(df.index, df['Employment_in_IT'], label='Bilişim (Gerçek)')
        ax_prophet.plot(prophet_forecast_services['ds'], prophet_forecast_services['yhat'], label='Prophet Hizmetler Tahmini', color='red')
        ax_prophet.plot(prophet_forecast_manufacturing['ds'], prophet_forecast_manufacturing['yhat'], label='Prophet İmalat Tahmini', color='blue')
        ax_prophet.plot(prophet_forecast_IT['ds'], prophet_forecast_IT['yhat'], label='Prophet Bilişim Tahmini', color='green')
        ax_prophet.fill_between(prophet_forecast_services['ds'], prophet_forecast_services['yhat_lower'], prophet_forecast_services['yhat_upper'], color='red', alpha=0.1)
        ax_prophet.fill_between(prophet_forecast_manufacturing['ds'], prophet_forecast_manufacturing['yhat_lower'], prophet_forecast_manufacturing['yhat_upper'], color='blue', alpha=0.1)
        ax_prophet.fill_between(prophet_forecast_IT['ds'], prophet_forecast_IT['yhat_lower'], prophet_forecast_IT['yhat_upper'], color='green', alpha=0.1)
        ax_prophet.set_xlabel('Tarih')
        ax_prophet.set_ylabel('İndeks')
        ax_prophet.set_title('Prophet Modeli ile Gelecek Tahminleri')
        ax_prophet.legend()
        st.pyplot(fig_prophet)

    elif sub_menu == 'Endeks Analizi':
        st.write("## Endeks Analizi")

        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        import streamlit as st

        # Excel dosyasını oku, başlıkları doğru şekilde ayarla
        df = pd.read_excel('imalat - hizmet - bilişim sektör endeks analiz verileri.xlsx', header=[0, 1, 2])


        # Başlıkları birleştir ve boş sütunları atla
        def clean_column_name(col):
            col_name = '_'.join(filter(None, map(str, col))).strip()
            col_name = col_name.replace('Unnamed:', '').replace('level_', '').replace('\n', ' ').strip()
            return col_name


        df.columns = [clean_column_name(col) for col in df.columns.values]
        df = df.loc[:, df.columns.notna()]

        # Sütun adlarını kontrol et
        print(df.columns)

        # Yıl ve ay sütunlarının başlıklarını düzenle
        df.rename(columns={
            '0_0_ 0_1_Year': 'Year',
            '1_0_ 1_1_Month': 'Month'
        }, inplace=True)

        # Tarih sütununu oluştur
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01', format='%Y-%m-%d',
                                    errors='coerce')

        # Eski yıl ve ay sütunlarını kaldırma
        df.drop(['Year', 'Month'], axis=1, inplace=True)

        # Tarih özellikleri ekleme
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        # Streamlit Başlangıç
        st.title('Sektör Endeksi Analiz ve Tahminler')

        # Zaman serisi grafiği
        st.write("## Zaman Serisi Grafiği")
        fig_ts, ax_ts = plt.subplots(figsize=(12, 6))
        df.plot(x='Date', y=['Services_Seasonal and calendar adjusted_Index',
                             'Information and communication_Seasonal and calendar adjusted_Index',
                             'Manufacturing_Seasonal and calendar adjusted_Index'], ax=ax_ts)
        ax_ts.set_xlabel('Date')
        ax_ts.set_ylabel('Index')
        ax_ts.set_title('Sector Indices Over Time')
        st.pyplot(fig_ts)

        # Eğitim ve test verisini ayırma
        features = ['Year', 'Month', 'Manufacturing_Seasonal and calendar adjusted_Index']
        target = 'Services_Seasonal and calendar adjusted_Index'
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Random Forest modelini oluştur ve eğit
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Tahminler
        df['RF_Predicted'] = rf_model.predict(X)

        # Gradient Boosting modelini oluştur ve eğit
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)

        # Tahminler
        df['GB_Predicted'] = gb_model.predict(X)

        # Genel Trendler
        st.write("## Genel Trendler")
        fig_trends, ax_trends = plt.subplots(figsize=(12, 6))
        df.set_index('Date')[['Services_Seasonal and calendar adjusted_Index', 'RF_Predicted', 'GB_Predicted']].plot(
            ax=ax_trends)
        ax_trends.set_xlabel('Date')
        ax_trends.set_ylabel('Index')
        ax_trends.set_title('Trendler ve Tahminler')
        st.pyplot(fig_trends)


        # Gelecek tahminlerini göstermek için tarih aralığını oluştur
        def forecast(model, feature_cols, years=10):
            future_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(months=1), periods=years * 12, freq='M')
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Year': future_dates.year,
                'Month': future_dates.month
            })
            future_df = future_df.set_index('Date')

            # Özelliklerin gerçekten var olduğundan emin olma
            for feature in feature_cols:
                if feature not in future_df.columns:
                    future_df[feature] = 0  # Yeni sütunları sıfır olarak başlat

            forecast = model.predict(future_df[feature_cols])
            future_df['Forecast'] = forecast
            return future_df


        # Gelecek tahminler
        future_df_rf = forecast(rf_model, features)
        future_df_gb = forecast(gb_model, features)

        # Gelecek tahminlerini grafiğe ekle
        st.write("## Gelecek 5-10 Yıl Tahminleri")
        fig_future, ax_future = plt.subplots(figsize=(12, 8))

        # Çizgi grafikleri için her iki modelin tahminlerini çiz
        ax_future.plot(future_df_rf.index, future_df_rf['Forecast'], label='Random Forest Forecast', color='blue')
        ax_future.plot(future_df_gb.index, future_df_gb['Forecast'], label='Gradient Boosting Forecast', color='green')
        ax_future.set_xlabel('Date')
        ax_future.set_ylabel('Index')
        ax_future.set_title('Future Sector Index Forecast')
        ax_future.legend()
        st.pyplot(fig_future)

    elif sub_menu == 'ithalat - ihracat analizi':
        st.write("## ithalat - ihracat analizi")


        # Streamlit uygulaması için başlık
        st.title('Gelecek Yıllara Dair İthalat-İhracat Tahmini')

        data_file_ihracat = 'streamlit kodlar toplam/nurdoğan/İhracat 2016-2022.xlsx'
        data_file_ithalat = 'streamlit kodlar toplam/nurdoğan/İthalat 2016-2022.xlsx'

        st.cache_data


        def load_data1():
            return pd.read_excel(data_file_ihracat)


        df1 = load_data1()

        st.cache_data


        def load_data2():
            return pd.read_excel(data_file_ithalat)


        df2 = load_data2()

        st.subheader('İhracat 2016-2022')
        st.write(df1)

        st.subheader('İthalat 2016-2022')
        st.write(df2)

        if st.button('İhracat Tahmini'):
            future_years = pd.DataFrame({'Year': [2023, 2024, 2025, 2026, 2027, 2028]})

            # Tahminler için veri çerçevesi
            predictions = future_years.copy()

            # Tahminlerin yapılması
            for column in df1.columns[1:]:
                X = df1[['Year']]
                y = df1[column]

                model = LinearRegression()
                model.fit(X, y)

                predictions[column] = model.predict(future_years)

            # Grafik oluşturma
            fig, ax = plt.subplots(figsize=(14, 10))

            # Mevcut verileri çizme
            for column in df1.columns[1:]:
                ax.plot(df1['Year'], df1[column], marker='o', label=f'{column} (Actual)')

            # Tahmin edilen verileri çizme
            for column in df1.columns[1:]:
                ax.plot(predictions['Year'], predictions[column], marker='o', linestyle='--',
                        label=f'{column} (Predicted)')

            ax.set_title('All Export Services Over Years with Predictions')
            ax.set_xlabel('Year')
            ax.set_ylabel('Value')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            # Grafik düzenini ayarlama
            plt.tight_layout()

            # Grafiği Streamlit uygulamasında gösterme
            st.pyplot(fig)

        if st.button('İthalat Tahmini'):
            future_years = pd.DataFrame({'Year': [2023, 2024, 2025, 2026, 2027, 2028]})

            # Tahminler için veri çerçevesi
            predictions = future_years.copy()

            # Tahminlerin yapılması
            for column in df2.columns[1:]:
                X = df2[['Year']]
                y = df2[column]

                model = LinearRegression()
                model.fit(X, y)

                predictions[column] = model.predict(future_years)

            # Grafik oluşturma
            fig, ax = plt.subplots(figsize=(14, 10))

            # Mevcut verileri çizme
            for column in df2.columns[1:]:
                ax.plot(df2['Year'], df2[column], marker='o', label=f'{column} (Actual)')

            # Tahmin edilen verileri çizme
            for column in df2.columns[1:]:
                ax.plot(predictions['Year'], predictions[column], marker='o', linestyle='--',
                        label=f'{column} (Predicted)')

            ax.set_title('All Import Services Over Years with Predictions')
            ax.set_xlabel('Year')
            ax.set_ylabel('Value')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            # Grafik düzenini ayarlama
            plt.tight_layout()

            # Grafiği Streamlit uygulamasında gösterme
            st.pyplot(fig)

    elif sub_menu == 'Ekonomik Faaliyetlere Göre Temel Göstergeler 2015-2022':
        st.write("## Ekonomik Faaliyetlere Göre Temel Göstergeler 2015-2022")

        import streamlit as st
        import pandas as pd
        import numpy as np
        import datetime as dt
        import matplotlib
        import matplotlib.pyplot as plt
        from kneed import KneeLocator

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.cluster import KMeans
        import seaborn as sns

        matplotlib.use('TkAgg')
        from yellowbrick.cluster import KElbowVisualizer
        import warnings

        warnings.filterwarnings("ignore")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        pd.set_option('display.width', 1000)

        st.header('Ekonomik Faaliyetlere Göre Temel Göstergeler 2015-2022')

        data_file = 'streamlit kodlar toplam/nurdoğan/123dataset.xlsx'

        st.cache_data


        def load_data():
            return pd.read_excel(data_file)


        df = load_data()

        st.subheader('Orijinal Dataset')
        st.write(df)
        merged_df = pd.DataFrame()

        # Her grup için aynı işlemi tekrarlayan bir döngü
        for i in range(0, 135, 9):
            temp_df = df.iloc[i:i + 9, 1:].T.iloc[1:]
            temp_df.columns = [
                'No of Enterprises', 'No of persons employed', 'No of employees', 'Personnel costs',
                'Turnover', 'Total purchases of goods and services', 'Change in stocks of goods and services',
                'Production value', 'Value added at factor costs'
            ]
            merged_df = pd.concat([merged_df, temp_df], ignore_index=False)

        merged_df = merged_df.apply(pd.to_numeric, errors='coerce')
        merged_df.reset_index(drop=True, inplace=True)

        if st.button('Orijinal Datasetle K-Means'):
            def split_dataframe(df, ranges):

                dfs = []
                for start, end in ranges:
                    dfs.append(df.loc[start:end])
                return dfs


            ranges = [(0, 6), (7, 13), (14, 20), (21, 27), (28, 34), (35, 41),
                      (42, 48), (49, 55), (56, 62), (63, 69), (70, 76), (77, 83),
                      (84, 90), (91, 97), (98, 104)]
            dfb1, dfc1, dfd1, dfe1, dff1, dfg1, dfh1, dfi1, dfj1, dfl1, dfm1, dfn1, dfp1, dfq1, dfr1 = split_dataframe(
                merged_df, ranges)

            dataframes = [dfb1, dfc1, dfd1, dfe1, dff1, dfg1, dfh1, dfi1, dfj1, dfl1, dfm1, dfn1, dfp1, dfq1, dfr1]

            # Her dataframe için log dönüşümünü ve min-max ölçeklendirmeyi uygulayalım
            scaled_dataframes = []
            for df in dataframes:
                # Log dönüşümü
                df_log = np.log1p(df)

                # Min-Max ölçekleme
                scaler = MinMaxScaler()
                df_scaled = df_log.copy()
                for col in df_scaled.columns:
                    df_scaled[[col]] = scaler.fit_transform(df_scaled[[col]])

                scaled_dataframes.append(df_scaled)


            def calculate_column_means(dataframes):
                means_list = []
                for df in dataframes:
                    means = df.mean().to_frame(name='Mean').T
                    means_list.append(means)
                return pd.concat(means_list, ignore_index=True)


            mean_df = calculate_column_means(scaled_dataframes)

            kmeans = KMeans(n_clusters=4, n_init=15, random_state=17).fit(mean_df)
            st.write(f"Küme Sayısı: {kmeans.n_clusters}")
            # st.write(f"Küme Merkezleri: {kmeans.cluster_centers_}")
            # st.write(f"Etiketler: {kmeans.labels_}")
            st.write(f"Inertia: {kmeans.inertia_}")

            kmeans = KMeans()
            ssd = []
            K = range(1, 15)
            for k in K:
                kmeans = KMeans(n_clusters=k).fit(mean_df)
                ssd.append(kmeans.inertia_)

            plt.figure()
            plt.plot(K, ssd, "bx-")
            plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
            plt.title("Optimum Küme Sayısı için Elbow Yöntemi")
            st.pyplot(plt)

            # Optimum K değeri için Elbow noktası belirleme
            knee_locator = KneeLocator(K, ssd, curve='convex', direction='decreasing')
            optimal_k = knee_locator.elbow
            st.write(f"Optimal Küme Sayısı: {optimal_k}")

            kmeans = KMeans(n_clusters=optimal_k).fit(mean_df)
            clusters_kmeans = kmeans.labels_

            mean_df['Cluster'] = clusters_kmeans
            sector_list = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'P', 'Q', 'R']
            mean_df['Sector'] = sector_list
            st.write("Kümeleme Sonuçları:")
            st.write(mean_df)

            fixed_size = 300
            df_final = pd.DataFrame(
                {'Cluster': clusters_kmeans, 'Sector': sector_list, 'Size': [fixed_size] * len(sector_list)})

            plt.figure(figsize=(14, 8))
            sns.scatterplot(x=df_final.index, y='Cluster', size='Size', hue='Sector', palette='tab10', data=df_final,
                            sizes=(fixed_size, fixed_size), alpha=0.7, legend=False)

            for i in range(len(df_final)):
                plt.text(df_final.index[i], df_final['Cluster'][i], df_final['Sector'][i],
                         ha='center', va='center', fontsize=9, color='black')

            plt.xticks(df_final.index, df_final['Sector'], rotation=90)
            plt.xlabel('Sector')
            plt.ylabel('Cluster')
            plt.title('Sector Clustering Visualization')
            plt.tight_layout()
            st.pyplot(plt)

        if st.button('Orijinal + Özellik Müh. Dataset K-Means'):
            merged_df = merged_df.drop(
                columns=['No of employees', 'Production value', 'Total purchases of goods and services',
                         'Change in stocks of goods and services'])

            merged_df['Empployed per Enterprises'] = merged_df['No of persons employed'] / merged_df[
                'No of Enterprises']
            merged_df['Person cost per Employed'] = merged_df['Personnel costs'] / merged_df['No of persons employed']
            merged_df['Turnover per Employed'] = merged_df['Turnover'] / merged_df['No of persons employed']
            merged_df['VAF per Employed'] = merged_df['Value added at factor costs'] / merged_df[
                'No of persons employed']
            merged_df['VAF per Turnover'] = merged_df['Value added at factor costs'] / merged_df['Turnover']
            merged_df.head()


            # st.subheader('Orijinal Kolonlar ve Yeni Türetilen Değişkenler')
            # st.write(merged_df)

            def split_dataframe(df, ranges):

                dfs = []
                for start, end in ranges:
                    dfs.append(df.loc[start:end])
                return dfs


            ranges = [(0, 6), (7, 13), (14, 20), (21, 27), (28, 34), (35, 41),
                      (42, 48), (49, 55), (56, 62), (63, 69), (70, 76), (77, 83),
                      (84, 90), (91, 97), (98, 104)]
            dfb1, dfc1, dfd1, dfe1, dff1, dfg1, dfh1, dfi1, dfj1, dfl1, dfm1, dfn1, dfp1, dfq1, dfr1 = split_dataframe(
                merged_df, ranges)

            dataframes = [dfb1, dfc1, dfd1, dfe1, dff1, dfg1, dfh1, dfi1, dfj1, dfl1, dfm1, dfn1, dfp1, dfq1, dfr1]

            # Her dataframe için log dönüşümünü ve min-max ölçeklendirmeyi uygulayalım
            scaled_dataframes = []
            for df in dataframes:
                # Log dönüşümü
                df_log = np.log1p(df)

                # Min-Max ölçekleme
                scaler = MinMaxScaler()
                df_scaled = df_log.copy()
                for col in df_scaled.columns:
                    df_scaled[[col]] = scaler.fit_transform(df_scaled[[col]])

                scaled_dataframes.append(df_scaled)


            def calculate_column_means(dataframes):
                means_list = []
                for df in dataframes:
                    means = df.mean().to_frame(name='Mean').T
                    means_list.append(means)
                return pd.concat(means_list, ignore_index=True)


            mean_df = calculate_column_means(scaled_dataframes)

            kmeans = KMeans(n_clusters=4, n_init=15, random_state=17).fit(mean_df)
            st.write(f"Küme Sayısı: {kmeans.n_clusters}")
            # st.write(f"Küme Merkezleri: {kmeans.cluster_centers_}")
            # st.write(f"Etiketler: {kmeans.labels_}")
            st.write(f"Inertia: {kmeans.inertia_}")

            kmeans = KMeans()
            ssd = []
            K = range(1, 15)
            for k in K:
                kmeans = KMeans(n_clusters=k).fit(mean_df)
                ssd.append(kmeans.inertia_)

            plt.figure()
            plt.plot(K, ssd, "bx-")
            plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
            plt.title("Optimum Küme Sayısı için Elbow Yöntemi")
            st.pyplot(plt)

            # Optimum K değeri için Elbow noktası belirleme
            knee_locator = KneeLocator(K, ssd, curve='convex', direction='decreasing')
            optimal_k = knee_locator.elbow
            st.write(f"Optimal Küme Sayısı: {optimal_k}")

            kmeans = KMeans(n_clusters=optimal_k).fit(mean_df)
            clusters_kmeans = kmeans.labels_

            mean_df['Cluster'] = clusters_kmeans
            sector_list = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'P', 'Q', 'R']
            mean_df['Sector'] = sector_list
            st.write("Kümeleme Sonuçları:")
            st.write(mean_df)

            fixed_size = 300
            df_final = pd.DataFrame(
                {'Cluster': clusters_kmeans, 'Sector': sector_list, 'Size': [fixed_size] * len(sector_list)})

            plt.figure(figsize=(14, 8))
            sns.scatterplot(x=df_final.index, y='Cluster', size='Size', hue='Sector', palette='tab10', data=df_final,
                            sizes=(fixed_size, fixed_size), alpha=0.7, legend=False)

            for i in range(len(df_final)):
                plt.text(df_final.index[i], df_final['Cluster'][i], df_final['Sector'][i],
                         ha='center', va='center', fontsize=9, color='black')

            plt.xticks(df_final.index, df_final['Sector'], rotation=90)
            plt.xlabel('Sector')
            plt.ylabel('Cluster')
            plt.title('Sector Clustering Visualization')
            plt.tight_layout()
            st.pyplot(plt)

        if st.button('Sadece Özellik Müh. Dataset K-Means'):
            merged_df = merged_df.drop(
                columns=['No of employees', 'Production value', 'Total purchases of goods and services',
                         'Change in stocks of goods and services'])

            merged_df['Empployed per Enterprises'] = merged_df['No of persons employed'] / merged_df[
                'No of Enterprises']
            merged_df['Person cost per Employed'] = merged_df['Personnel costs'] / merged_df['No of persons employed']
            merged_df['Turnover per Employed'] = merged_df['Turnover'] / merged_df['No of persons employed']
            merged_df['VAF per Employed'] = merged_df['Value added at factor costs'] / merged_df[
                'No of persons employed']
            merged_df['VAF per Turnover'] = merged_df['Value added at factor costs'] / merged_df['Turnover']
            merged_df.head()

            # Orjinal Columnları dropluyoruz
            merged_df = merged_df.drop(
                columns=['No of Enterprises', 'No of persons employed', 'Personnel costs', 'Turnover',
                         'Value added at factor costs'])


            # st.subheader('Özellik Mühendisliği Uygulanmış')
            # st.write(merged_df)

            def split_dataframe(df, ranges):

                dfs = []
                for start, end in ranges:
                    dfs.append(df.loc[start:end])
                return dfs


            ranges = [(0, 6), (7, 13), (14, 20), (21, 27), (28, 34), (35, 41),
                      (42, 48), (49, 55), (56, 62), (63, 69), (70, 76), (77, 83),
                      (84, 90), (91, 97), (98, 104)]
            dfb1, dfc1, dfd1, dfe1, dff1, dfg1, dfh1, dfi1, dfj1, dfl1, dfm1, dfn1, dfp1, dfq1, dfr1 = split_dataframe(
                merged_df, ranges)

            dataframes = [dfb1, dfc1, dfd1, dfe1, dff1, dfg1, dfh1, dfi1, dfj1, dfl1, dfm1, dfn1, dfp1, dfq1, dfr1]

            # Her dataframe için log dönüşümünü ve min-max ölçeklendirmeyi uygulayalım
            scaled_dataframes = []
            for df in dataframes:
                # Log dönüşümü
                df_log = np.log1p(df)

                # Min-Max ölçekleme
                scaler = MinMaxScaler()
                df_scaled = df_log.copy()
                for col in df_scaled.columns:
                    df_scaled[[col]] = scaler.fit_transform(df_scaled[[col]])

                scaled_dataframes.append(df_scaled)


            def calculate_column_means(dataframes):
                means_list = []
                for df in dataframes:
                    means = df.mean().to_frame(name='Mean').T
                    means_list.append(means)
                return pd.concat(means_list, ignore_index=True)


            mean_df = calculate_column_means(scaled_dataframes)

            kmeans = KMeans(n_clusters=4, n_init=15, random_state=17).fit(mean_df)
            st.write(f"Küme Sayısı: {kmeans.n_clusters}")
            # st.write(f"Küme Merkezleri: {kmeans.cluster_centers_}")
            # st.write(f"Etiketler: {kmeans.labels_}")
            st.write(f"Inertia: {kmeans.inertia_}")

            kmeans = KMeans()
            ssd = []
            K = range(1, 15)
            for k in K:
                kmeans = KMeans(n_clusters=k).fit(mean_df)
                ssd.append(kmeans.inertia_)

            plt.figure()
            plt.plot(K, ssd, "bx-")
            plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
            plt.title("Optimum Küme Sayısı için Elbow Yöntemi")
            st.pyplot(plt)

            # Optimum K değeri için Elbow noktası belirleme
            knee_locator = KneeLocator(K, ssd, curve='convex', direction='decreasing')
            optimal_k = knee_locator.elbow
            st.write(f"Optimal Küme Sayısı: {optimal_k}")

            kmeans = KMeans(n_clusters=optimal_k).fit(mean_df)
            clusters_kmeans = kmeans.labels_

            mean_df['Cluster'] = clusters_kmeans
            sector_list = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'P', 'Q', 'R']
            mean_df['Sector'] = sector_list
            st.write("Kümeleme Sonuçları:")
            st.write(mean_df)

            fixed_size = 300
            df_final = pd.DataFrame(
                {'Cluster': clusters_kmeans, 'Sector': sector_list, 'Size': [fixed_size] * len(sector_list)})

            plt.figure(figsize=(14, 8))
            sns.scatterplot(x=df_final.index, y='Cluster', size='Size', hue='Sector', palette='tab10', data=df_final,
                            sizes=(fixed_size, fixed_size), alpha=0.7, legend=False)

            for i in range(len(df_final)):
                plt.text(df_final.index[i], df_final['Cluster'][i], df_final['Sector'][i],
                         ha='center', va='center', fontsize=9, color='black')

            plt.xticks(df_final.index, df_final['Sector'], rotation=90)
            plt.xlabel('Sector')
            plt.ylabel('Cluster')
            plt.title('Sector Clustering Visualization')
            plt.tight_layout()
            st.pyplot(plt)

        st.markdown("""
        # SECTORS
        - **B** - Mining and quarrying
        - **C** - Manufacture
        - **D** - Electricity, gas, steam and air conditioning supply
        - **E** - Water supply; sewerage, waste management and remediation
        - **F** - Construction
        - **G** - Wholesale and retail trade; repair of motor vehicles and motorcycles
        - **H** - Transportation and storage
        - **I** - Accommodation and food service activities
        - **J** - Information and communication
        - **L** - Real estate activities
        - **M** - Professional, scientific and technical activities
        - **N** - Administrative and support service activities
        - **P** - Education
        - **Q** - Human health and social work activities
        - **R** - Arts, entertainment and recreation
        """)

    elif sub_menu == 'Bilanço Analizi':
        st.write("## Bilanço Analizi")
        import pandas as pd
        import streamlit as st
        import numpy as np
        import matplotlib.pyplot as plt
        import warnings
        from sklearn.linear_model import LinearRegression

        warnings.filterwarnings("ignore")

        # Streamlit başlığı
        st.title("Bilanço Analizi")

        # Veriyi yükleme (CSV dosyanızın yolu burada)
        data_file = 'streamlit kodlar toplam/sencer/bilanco_verileri_duzenli (1).csv'


        @st.cache_data
        def load_data(file_path):
            df = pd.read_csv(file_path)
            columns_to_drop = ['Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21']
            df = df.drop(columns=columns_to_drop)
            df.iloc[:, 0] = df.iloc[:, 0].fillna(method='ffill')
            df.iloc[:, 0] = df.iloc[:, 0].str.strip()
            df.iloc[:, 1] = df.iloc[:, 1].str.strip()
            return df


        # Veriyi yükle
        df = load_data(data_file)

        # Veriyi göster
        st.write("Veri:")
        st.write(df.head(50))

        # Brüt satışlar (Gross sales) satırlarını filtrele
        gross_sales_df = df[df.iloc[:, 1] == "A-Gross sales"]

        # Yıl sütunlarını belirle (gerçek yıl sütun aralığına göre ayarla)
        year_columns = gross_sales_df.columns[2:16]

        # Yıl sütunlarını sayısal değerlere dönüştür, virgül ve boşlukları kaldır
        gross_sales_df[year_columns] = gross_sales_df[year_columns].replace({' ': '', ',': ''}, regex=True).astype(
            float)

        # Her yıl için ortalama brüt satışları hesapla
        average_sales_per_year = gross_sales_df[year_columns].mean()

        # Normalize edilmiş yüzdelik değerleri hesapla (tüm yılların ortalamasına göre)
        base_value = average_sales_per_year.mean()  # Tüm yılların ortalama değeri
        normalized_percentages = (average_sales_per_year / base_value) * 100

        # Yılları ve normalize edilmiş yüzdelik değerleri içeren bir DataFrame oluştur
        normalized_scores_data = pd.DataFrame({
            'Year': year_columns,
            'Normalized Percentage': normalized_percentages.values
        })

        # Normalize edilmiş yüzdelik değerleri göster
        st.write("Normalize Edilmiş Yüzdelik Değerler:")
        st.write(normalized_scores_data)

        # Normalize edilmiş yüzdelik değerleri kullanarak grafik oluştur
        st.write("Normalize Edilmiş Yüzdelik Değerler Grafiği:")
        fig, ax = plt.subplots()
        ax.plot(normalized_scores_data['Year'], normalized_scores_data['Normalized Percentage'], marker='o')
        ax.set_title('Yıllık Brüt Satışların Normalize Edilmiş Yüzdelik Değerleri (Ortalama Değere Göre)')
        ax.set_xlabel('Yıl')
        ax.set_ylabel('Normalize Edilmiş Yüzde')
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)


        # Belirli bir yıl için en çok kazandıran sektörü bulmak için fonksiyon
        def analyze_year(df, year):
            year_column = str(year)
            sector_sales = df[[df.columns[0], year_column]].copy()
            sector_sales[year_column] = sector_sales[year_column].fillna(0)
            sector_sales_sorted = sector_sales.sort_values(by=year_column, ascending=False)
            top_sector = sector_sales_sorted.iloc[0]
            return sector_sales_sorted, top_sector


        # Tüm yıllar için analizi tekrarla ve sonuçları göster
        for year in year_columns:
            sector_sales_sorted, top_sector = analyze_year(gross_sales_df, year)
            st.write(f"Sektörlerin {year} yılı brüt satışlarına göre sıralaması:")
            st.write(sector_sales_sorted)
            st.write(
                f"{year} yılında en çok kazandıran sektör: Sektör: {top_sector[df.columns[0]]}, {year} Yılı Brüt Satış: {top_sector[year]}")

        # Geçmiş yıllar için brüt satışları göster
        past_years = np.array([int(year) for year in year_columns])
        past_predictions = pd.DataFrame({'Year': past_years})

        sectors = gross_sales_df.iloc[:, 0].unique()
        for sector in sectors:
            sector_data = gross_sales_df[gross_sales_df.iloc[:, 0] == sector]
            sector_sales = sector_data[year_columns].values.flatten()
            sector_sales = np.nan_to_num(sector_sales)
            past_predictions[sector] = sector_sales

        st.write("Geçmiş Yıllar İçin Sektör Brüt Satışları:")
        fig, ax = plt.subplots(figsize=(18, 8))
        for sector in sectors:
            ax.plot(past_predictions['Year'], past_predictions[sector], marker='o', label=sector)
        ax.set_title('Geçmiş Yıllar İçin Sektör Brüt Satışları')
        ax.set_xlabel('Yıl')
        ax.set_ylabel('Brüt Satış')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Gelecekteki yıllar için tahminler
        future_years = np.arange(2023, 2033)
        past_years = past_years.reshape(-1, 1)
        future_predictions = pd.DataFrame({'Year': future_years})

        for sector in sectors:
            sector_data = gross_sales_df[gross_sales_df.iloc[:, 0] == sector]
            sector_sales = sector_data[year_columns].values.flatten()
            sector_sales = np.nan_to_num(sector_sales)
            model = LinearRegression()
            model.fit(past_years, sector_sales)
            future_sales = model.predict(future_years.reshape(-1, 1))
            future_predictions[sector] = future_sales

        # Gelecekteki yıllar için tahminleri grafikle göster
        st.write("Önümüzdeki 10 Yıl İçin Sektör Tahminleri:")
        fig, ax = plt.subplots(figsize=(10, 6))
        for sector in sectors:
            ax.plot(future_predictions['Year'], future_predictions[sector], marker='o', label=sector)
        ax.set_title('Önümüzdeki 10 Yıl İçin Sektör Tahminleri')
        ax.set_xlabel('Yıl')
        ax.set_ylabel('Tahmini Brüt Satış')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    elif sub_menu == 'Startup Yatırım Analizi':
        st.write("## EStartup Yatırım Analizi")
        import streamlit as st
        import pandas as pd
        import matplotlib.pyplot as plt
        import warnings
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        warnings.filterwarnings("ignore")

        # Streamlit başlığı
        st.title("Startup Yatırım Analizi")

        # Veriyi yükleme (CSV dosyanızın yolu burada)
        data_file = 'streamlit kodlar toplam/sencer/investments_VC-Backup.csv'


        @st.cache_data
        def load_data(file_path):
            # Veriyi yükleme
            data_csv = pd.read_csv(file_path, encoding='latin1')

            # Kullanılmayan sütunları düşürme
            columns_to_drop = [
                'state_code', 'round_A', 'round_B', 'round_C', 'round_D', 'round_E', 'round_F',
                'round_G', 'round_H', 'founded_at', 'founded_month', 'founded_quarter',
                'founded_year', 'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',
                'private_equity', 'post_ipo_equity', 'post_ipo_debt', 'secondary_market',
                'product_crowdfunding', 'equity_crowdfunding'
            ]
            data = data_csv.drop(columns=columns_to_drop)

            # Sütun isimlerinden boşlukları temizleme
            data.columns = data.columns.str.strip()

            # Para birimini temizleme ve dönüşüm
            data['funding_total_usd'] = pd.to_numeric(data['funding_total_usd'].replace('[\\$,]', '', regex=True),
                                                      errors='coerce')

            # Tarih dönüşümleri
            data['first_funding_at'] = pd.to_datetime(data['first_funding_at'], errors='coerce')
            data['last_funding_at'] = pd.to_datetime(data['last_funding_at'], errors='coerce')

            # Boş değerleri temizleme
            data = data.dropna(subset=['funding_total_usd', 'funding_rounds'])

            # Negatif değerleri ve sıfırdan küçük değerleri temizleme
            data = data[data['funding_total_usd'] >= 0.1e8]

            return data


        # Veriyi yükle
        data = load_data(data_file)

        # Veriyi göster
        st.write("Veri:")
        st.write(data.head(60))

        # Ülkelere göre ortalama yatırım miktarı
        avg_funding_by_country = data.groupby('country_code')['funding_total_usd'].mean().sort_values(ascending=False)

        st.write("Ülkelere Göre Ortalama Yatırım Miktarı:")
        fig, ax = plt.subplots(figsize=(24, 8))
        avg_funding_by_country.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Average Funding Total by Country')
        ax.set_xlabel('Country Code')
        ax.set_ylabel('Average Funding Total (USD)')
        ax.grid(True)
        st.pyplot(fig)

        # 2000 yılı sonrası yılları filtreleme
        data = data[data['first_funding_at'].dt.year > 2000]


        def plot_funding_by_sector_per_year(data, year):
            yearly_data = data[data['first_funding_at'].dt.year == year]
            total_funding_by_sector = yearly_data.groupby('market')['funding_total_usd'].sum().sort_values(
                ascending=False)
            fig, ax = plt.subplots(figsize=(24, 8))
            total_funding_by_sector.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f'Total Funding Amount by Sector in {year}')
            ax.set_xlabel('Sector')
            ax.set_ylabel('Total Funding Amount (USD)')
            ax.grid(True)
            st.pyplot(fig)


        # 2000 yılı sonrasını gösteren plotları bas
        for year in range(2001, data['first_funding_at'].dt.year.max() + 1):
            st.write(f"{year} Yılında Sektöre Göre Toplam Yatırım Miktarı:")
            plot_funding_by_sector_per_year(data, year)

    if menu == 'Start-Up Fikri':
        st.header('Start-Up Fikri Önerileri')

        # İstihdam ve endeks analizlerinin sonuçlarını yükleyin
        st.subheader('İstihdam Analizi Sonuçları')
        st.write('İstihdam verilerine göre büyüyen sektörler:')
        st.image('istihdam_trendleri.png')

        st.subheader('Endeks Analizi Sonuçları')
        st.write('Endeks verilerine göre yüksek performans gösteren sektörler:')
        st.image('endeks_trendleri.png')

        # Start-Up Fikri Önerisi
        st.write(
            """
            İstihdam ve endeks analizlerine dayanarak, aşağıdaki start-up fikri önerilerini dikkate alabilirsiniz:

            1. **Yüksek Büyüme Gösteren Sektörlerde Yatırım**: Teknoloji ve yeşil enerji sektörlerinde start-up oluşturma.
            2. **Yükselen Trendleri Takip Etme**: Endekslerde hızlı artış gösteren sektörlerde yenilikçi çözümler geliştirme.
            3. **Verimlilik Araçları Geliştirme**: Sektörlerdeki verimlilik artırıcı çözümler sunma.
            """
        )

if __name__ == "__main__":
    main()
