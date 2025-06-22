import streamlit as st
import geopandas as gpd # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.cluster.hierarchy as sch
import folium # type: ignore
from folium import Tooltip # type: ignore
from streamlit_folium import st_folium # type: ignore

st.set_page_config(page_title="Clustering App", layout="wide", page_icon="🏫")

st.title("Aplikasi Clustering Tingkat Pendidikan Kabupaten Bangkalan 📖")

uploaded_file = st.file_uploader("📂 Upload Dataset CSV", type="csv")

if uploaded_file:
    dfdesa = pd.read_csv(uploaded_file)
    st.header("📋 Data Awal")
    st.dataframe(dfdesa.head(), hide_index=True)

    # Hapus kolom
    kolom_dihapus = ["No.", "JML Penduduk", "JML Penduduk Belum/Tidak Bekerja", "Kab/Kota", "Kecamatan"]
    df_cleaned = dfdesa.drop(columns=kolom_dihapus, errors='ignore')

    st.header("📋 Data Fitur yang Digunakan")
    st.dataframe(df_cleaned.head(), hide_index=True)

    st.header("🔍 Pengecekan Missing Value")
    st.dataframe(data=df_cleaned.isnull().sum(), column_config={'_index': 'Fitur', '0': 'Jumlah Missing Value'})

    # Ganti missing value dengan nilai mean
    numerik_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[numerik_columns] = df_cleaned[numerik_columns].fillna(df_cleaned[numerik_columns].mean())

    st.header("📋 Data Setelah Penanganan Missing value")
    st.dataframe(df_cleaned.head(), hide_index=True)

    st.header("🔍 Pengecekan Data Duplikat")
    st.metric("Jumlah Data Duplikat", df_cleaned.duplicated().sum())

    # Hapus nilai duplikat
    df_cleaned = df_cleaned.drop_duplicates()

    st.header("📋 Data Setelah Data Duplikat Dihapus")
    st.dataframe(df_cleaned.head(), hide_index=True)

    st.header("🔍 Pengecekan Outliers")
    fig, ax = plt.subplots(figsize=(25, 10))

    dfdesa.boxplot(column=["TIDAK SEKOLAH", "BELUM TAMAT SD", "SD", "SLTP", "SLTA",
       "DI/DII", "DIII", "DIV/S1", "S2", "S3", "SD/SEDERAJAT", "SMP/SEDERAJAT",
       "SMA/SEDERAJAT", "SMK/SEDERAJAT", "PERGURUAN TINGGI"])
    st.pyplot(fig)

    # Capping outliers
    def cap_outliers_all_columns(df):
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower, lower, df[col])
            df[col] = np.where(df[col] > upper, upper, df[col])
        return df

    dfdesacleaned = cap_outliers_all_columns(df_cleaned)

    st.header("📋 Data Setelah Data yang Outlier Ditangani")
    fig, ax = plt.subplots(figsize=(25, 10))

    dfdesacleaned.boxplot(column=["TIDAK SEKOLAH", "BELUM TAMAT SD", "SD", "SLTP", "SLTA",
       "DI/DII", "DIII", "DIV/S1", "S2", "S3", "SD/SEDERAJAT", "SMP/SEDERAJAT",
       "SMA/SEDERAJAT", "SMK/SEDERAJAT", "PERGURUAN TINGGI"])
    st.pyplot(fig)

    st.header("📋 Data Setelah Dinormalisasi")
    # Normalisasi
    fitur = ["TIDAK SEKOLAH", "BELUM TAMAT SD", "SD", "SLTP", "SLTA",
       "DI/DII", "DIII", "DIV/S1", "S2", "S3", "SD/SEDERAJAT", "SMP/SEDERAJAT",
       "SMA/SEDERAJAT", "SMK/SEDERAJAT", "PERGURUAN TINGGI"]
    x_train = dfdesacleaned[fitur].values
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    df_normalisasi = pd.DataFrame(x_train_scaled, columns=fitur)
    st.dataframe(df_normalisasi.head(), hide_index=True)

    # Clustering
    st.header("⚙️ Pilih Metode Clustering")
    metode = st.selectbox("Metode:", ["K-Means", "Hierarchical Ward"])

    if metode == "K-Means":
        silhouette_scores = []
        dbi_scores = []

        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters_km = kmeans.fit_predict(x_train_scaled)

            # Menghitung Silhouette Score
            silhouette_avg_km = silhouette_score(x_train_scaled, clusters_km)
            silhouette_scores.append(silhouette_avg_km)

            # Menghitung Davies-Bouldin Index
            dbi_avg_km = davies_bouldin_score(x_train_scaled, clusters_km)
            dbi_scores.append(dbi_avg_km)

        # Elbow Plot
        st.header("📈 Elbow Plot")
        inertias = []
        k_range = range(1, 11)
        for i in k_range:
            km = KMeans(n_clusters=i, random_state=42).fit(x_train_scaled)
            inertias.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, inertias, marker='o')
        ax.set_xlabel("Jumlah Cluster")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")

        st.pyplot(fig)
        
        k = st.slider("Pilih jumlah cluster", 2, 10, 2)
        model = KMeans(n_clusters=k, random_state=42)
        cluster = model.fit_predict(x_train_scaled)
        dfdesa["Cluster"] = cluster

    elif metode == "Hierarchical Ward":
        silhouette_scores = []
        dbi_scores = []

        for n_clusters in range(2, 11):
            hie = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            clusters_hie = hie.fit_predict(x_train_scaled)

            # Menghitung Silhouette Score
            silhouette_avg_hie = silhouette_score(x_train_scaled, clusters_hie)
            silhouette_scores.append(silhouette_avg_hie)

            # Menghitung Davies-Bouldin Index
            dbi_avg_hie = davies_bouldin_score(x_train_scaled, clusters_hie)
            dbi_scores.append(dbi_avg_hie)

        # Dendrogram
        st.header("🧬 Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 5))
        dendro = sch.dendrogram(sch.linkage(x_train_scaled, method="ward"), ax=ax)
        st.pyplot(fig)

        k = st.slider("Pilih jumlah cluster", 2, 10, 2)
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        cluster = model.fit_predict(x_train_scaled)
        dfdesa["Cluster"] = cluster

    # Evaluasi
    st.header("📊 Evaluasi")

    # Silhouette dan Davies-Bouldin Index terbaik
    col1, col2 =st.columns(2)

    with col1:
        sil = round(silhouette_score(x_train_scaled, cluster), 4)
        st.metric(label="Silhouette Score", value=sil)

    with col2:
        dbi = round(davies_bouldin_score(x_train_scaled, cluster), 4)
        st.metric(label="Davies-Bouldin Index", value=dbi)

    # Nilai silhouette dan Davies-Bouldin Index dari 2 cluster hingga 10 cluster
    df_scores = pd.DataFrame({
        "Silhouette Score": silhouette_scores, 
        "Davies-Bouldin Index": dbi_scores
    }, index=range(2, 11))

    def highlight_max_min(s):
        if s.name == "Silhouette Score":
            return ["background-color: #5DE2E7" if v == s.max() else '' for v in s]
        elif s.name == "Davies-Bouldin Index":
            return ["background-color: #FE9900" if v == s.min() else '' for v in s] 
        
    st.dataframe(df_scores.style.apply(highlight_max_min), column_config={"_index": "Jumlah Cluster"})

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    # Plot Silhouette Scores
    ax[0].plot(range(2, 11), silhouette_scores, marker='o', color='#5DE2E7', label='Silhouette Score')
    ax[0].set_title('Silhouette Score untuk Berbagai Jumlah Cluster', pad=10)
    ax[0].set_xlabel('Jumlah Cluster')
    ax[0].set_ylabel('Silhouette Score')
    ax[0].tick_params(axis='x', labelsize=12)
    ax[0].grid(True)

    # Plot Davies-Bouldin Index
    ax[1].plot(range(2, 11), dbi_scores, marker='o', color='#FE9900', label='Davies-Bouldin Index')
    ax[1].set_title('Davies-Bouldin Index untuk Berbagai Jumlah Cluster', pad=10)
    ax[1].set_xlabel('Jumlah Cluster')
    ax[1].set_ylabel('Davies-Bouldin Index')
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].grid(True)

    st.pyplot(fig)

    # Analisis
    st.header("📊 Analisis")
    fitur = ['TIDAK SEKOLAH', 'BELUM TAMAT SD', 'SD', 'SLTP', 'SLTA',
         'DI/DII', 'DIII', 'DIV/S1', 'S2', 'S3', 'SD/SEDERAJAT', 'SMP/SEDERAJAT',
         'SMA/SEDERAJAT', 'SMK/SEDERAJAT', 'PERGURUAN TINGGI']
    colors = ['#E4080A', '#FFDE59', '#7DDA58', '#5DE2E7', '#EFC3CA', '#CC6CE7',
            '#57ABAF', '#BFD641', '#98F5F9', '#FE9900', '#EFC3CA', '#FFECA1',
            '#709E5E', '#8D6F64', '#CECECE']

    # Menghitung persentase setiap fitur dalam setiap cluster
    st.subheader("📊 Persentase Setiap Fitur dalam Setiap Cluster")
    kolom_sum = dfdesa.groupby(by='Cluster')[fitur].sum()
    kolom_percent = kolom_sum.div(kolom_sum.sum(axis=1), axis=0) * 100
    percentase = kolom_percent.round(4)
    total = kolom_percent.sum(axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    ax[0].pie(x=percentase.loc[0], labels=fitur, autopct='%1.1f%%', colors=colors)
    ax[0].set_title('Cluster 0', fontsize=15)

    ax[1].pie(x=percentase.loc[1], labels=fitur, autopct='%1.1f%%', colors=colors)
    ax[1].set_title('Cluster 1', fontsize=15)

    st.pyplot(fig)

    percentase_total = percentase.copy().astype('str') + '%'
    percentase_total['Total'] = total.round(1).astype(str) + '%'
    st.write(percentase_total)

    # Clustering per Kecamatan
    st.subheader("📊 Persebaran Cluster per Kecamatan")

    kecamatan = st.selectbox(
        "Masukkan Nama Kecamatan", 
        (
            "BANGKALAN",
            "SOCAH",
            "BURNEH",
            "KAMAL",
            "AROSBAYA",
            "GEGER",
            "KLAMPIS",
            "SEPULU",
            "TANJUNG BUMI",
            "KOKOP",
            "KWANYAR",
            "LABANG",
            "TANAH MERAH",
            "TRAGAH",
            "BLEGA",
            "MODUNG",
            "KONANG",
            "GALIS",
        )
    )
    byKecamatan = dfdesa[dfdesa['Kecamatan'] == kecamatan]
    byKecamatandf = byKecamatan[['Kecamatan', 'Desa', 'Cluster']].sort_values(by='Cluster')

    st.dataframe(byKecamatandf, hide_index=True)

    # Jumlah tiap cluster per kecamatan
    st.subheader("📊 Jumlah Tiap Cluster per Kecamatan")
    jml_cluster = byKecamatandf.groupby(by='Cluster').Desa.count()
    percentase_per_cluster = (jml_cluster/jml_cluster.sum())*100

    hasil_df = pd.DataFrame({
        'Jumlah Desa': jml_cluster,
        'Persentase (%)': percentase_per_cluster.round(2).astype(str) + '%'
    })
    st.dataframe(hasil_df, column_config={"_index": "Cluster"})

    # Menampilkan hasil clustering
    st.header("📊 Hasil Clustering")

    # Warna untuk cluster
    colors = ['#66c2a5', '#ffd92f']
    labels = ['0', '1']

    clust = st.number_input("Pilih Cluster", value=0)

    df = pd.DataFrame({
        "Jumlah Desa": dfdesa['Cluster'].value_counts(),
        "Persentase (%)": (dfdesa['Cluster'].value_counts() / dfdesa['Cluster'].value_counts().sum() * 100).round(2).astype(str) + '%'
    })
    
    if clust == 0 or clust == 1:
        df_filtered = dfdesa[dfdesa['Cluster'] == clust]
        st.dataframe(df_filtered)
        st.dataframe(df, column_config={"_index": "Cluster"})
    else:
        st.badge(":orange-badge ⚠️ Cluster tidak ditemukan")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(
        x=dfdesa['Cluster'].value_counts().index.astype(str), 
        height=dfdesa['Cluster'].value_counts(),
        color=colors
    )
    ax.set_title('Jumlah Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Jumlah')

    st.pyplot(fig)

    # Menampilkan peta
    st.header("🌏Peta Bangkalan")

    if "show_map" not in st.session_state:
        st.session_state.show_map = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Tampilkan Peta"):
            st.session_state.show_map = True
    with col2:
        if st.button("Sembunyikan Peta"):
            st.session_state.show_map = False

    if st.session_state.show_map:
        # Baca shapefile
        gdf_all = gpd.read_file("gadm36_IDN_4/gadm36_IDN_4.shp")

        # Filter Bangkalan
        gdf_bangkalan = gdf_all[gdf_all["NAME_2"] == "Bangkalan"]

        # Uppercase nama kecamatan dan desa
        dfdesa["Kecamatan"] = dfdesa["Kecamatan"].str.upper().str.strip()
        dfdesa["Desa"] = dfdesa["Desa"].str.upper().str.strip()
        gdf_bangkalan["NAME_3"] = gdf_bangkalan["NAME_3"].str.upper().str.strip()
        gdf_bangkalan["NAME_4"] = gdf_bangkalan["NAME_4"].str.upper().str.strip()

        # Merge shapefile dengan hasil clustering
        gdf_joined = gdf_bangkalan.merge(
            dfdesa,
            left_on=["NAME_3", "NAME_4"],
            right_on=["Kecamatan", "Desa"]
        )

        # Hasil merge
        st.write("Jumlah Desa:", len(gdf_joined), "Desa")

        # Validasi geometry
        gdf_joined = gdf_joined[gdf_joined.geometry.notnull() & ~gdf_joined.geometry.is_empty]
        gdf_joined = gdf_joined[gdf_joined.is_valid]

        # Set CRS jika belum ada
        if gdf_joined.crs is None:
            gdf_joined.set_crs(epsg=4326, inplace=True)
        else:
            gdf_joined = gdf_joined.to_crs(epsg=4326)

        # Ambil titik tengah peta
        center = gdf_joined.geometry.centroid.unary_union.centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=10, tiles='cartodbpositron')

        # Tambahkan layer per desa
        for _, row in gdf_joined.iterrows():
            geo_json = gpd.GeoSeries([row.geometry]).__geo_interface__

            folium.GeoJson(
                data=geo_json,
                style_function=lambda feature, color=colors[row["Cluster"]]: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.7,
                },
                tooltip=Tooltip(f"Desa: {row['Desa'].title()}<br>Kecamatan: {row['Kecamatan'].title()}<br>Cluster: {row['Cluster']}")
            ).add_to(m)

        # Tambahkan legend
        legend_html = """
        <style>
        .legend {
            position: absolute;
            bottom: 50px;
            left: 10px;
            z-index: 9999;
            background-color: white;
            border: 2px solid gray;
            padding: 10px;
            font-size: 14px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        }
        .legend i {
            width: 14px;
            height: 14px;
            float: left;
            margin-right: 8px;
            opacity: 0.7;
        }
        </style>

        <div class='legend'>
            <b>Legend:</b><br>
            <i style='background: #66c2a5'></i> Cluster 0<br>
            <i style='background: #ffd92f'></i> Cluster 1<br>
        </div>
        """
        
        # Menampilkan peta
        st_folium(m, width=1000, height=600)

        # Menampilkan legend
        st.markdown(legend_html, unsafe_allow_html=True)
    else:
        st.info("Tekan tombol 'Tampilkan Peta' untuk melihat peta hasil clustering")

    # Download hasil
    csv = dfdesa.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Hasil CSV", csv, "hasil_clustering.csv", "text/csv")

st.caption("by M. Nur Syamsi Maulidi")
