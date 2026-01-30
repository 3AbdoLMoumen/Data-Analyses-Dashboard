# 📊 Data-Analyses-Dashboard

This project is a **data visualization and analysis dashboard** that allows you to upload any dataset (CSV) and interactively explore it using **automatic graph suggestions, statistical analysis, and machine learning tools**. 
 
https://data-analyses-dashboard.streamlit.app/

### 🚀 Key Features
- **Automatic Graph Suggestion** – The dashboard suggests the best visualization based on your selected variables  
- **Adaptable to Any Dataset** – Works with numerical, categorical, or mixed data  
- **Statistical Parameters** – Computes central tendency (mean, median) and dispersion (std, variance)  
- **Numerical Binning** – Option to convert numerical features into categorical classes  
- **Interactive Graphs** – Histogram, Scatter, Bar, Box, Count, Pie, and Pair Plots  
- **3D Scatter Plots** 
- **Machine Learning Features** - Optional Unsupervised learning 'clustering (KMeans) or anomaly detection' Integrated with the Scatter Plot
- **Correlation Analysis** – Quickly check correlation between numerical variables  
- **Exportable** – Save graphs and tables for reports  

### 💻 Installation
```bash
git clone https://github.com/3AbdoLMoumen/Data-Analyses-Dashboard
cd Data-Analyses-Dashboard
pip install -r requirements.txt
```
### ▶️ Usage
```bash
streamlit run app.py
```

![Dashboard Demo](Demo.gif)

- Upload your CSV file 📁
- Dashboard automatically detects numerical and categorical columns
- Select X and Y variables (Z for 3D plots optional)
- View suggested chart types or manually select one 📊
- Explore correlations, statistical parameters, and pair plots 🔍
- Export charts and tables for reporting 💾

### 💡 Future Works
- Adding more ML features (classification; regressions...)
- Improving Application Efficency
- Adding LLM powered Data Interpretation
