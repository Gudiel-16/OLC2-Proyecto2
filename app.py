from cgitb import text
import streamlit as st
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from openpyxl import Workbook as wb

def main():

    try:

        st.title("Machine Learning")

        menu_archivos = ["CSV", "XLS", "XLSX", "JSON"]
        op_archivo = st.sidebar.selectbox("ARCHIVO", menu_archivos)

        menu_algoritmo = ["Regresion Lineal", "Regresion Polinomial", "Clasificador Gaussiano", "Arboles de decision", "Redes Neuronales"]
        op_algoritmo = st.sidebar.selectbox("ALGORITMO", menu_algoritmo)

        if op_archivo == "CSV":
            st.subheader("Seleccionar archivo CSV")
            data_file_csv = st.file_uploader("Subir archivo", type=["CSV"])
            if data_file_csv is not None:
                #st.write(type(data_file_csv))
                #file_details = {"Nombre": data_file_csv.name, "Tipo": data_file_csv.type, "Tamanio": data_file_csv.size}
                #st.write(file_details)
                df = pd.read_csv(data_file_csv)
                st.dataframe(df.head())            
                ejecutar(df, op_algoritmo)

        elif op_archivo == "XLS":
            st.subheader("Seleccionar archivo XLS")
            data_file_xls = st.file_uploader("Subir archivo", type=["xls"])
            if data_file_xls is not None:
                df = pd.read_excel(data_file_xls)
                st.dataframe(df.head())
                ejecutar(df, op_algoritmo)
        
        elif op_archivo == "XLSX":
            st.subheader("Seleccionar archivo XLSX")
            data_file_xlsx = st.file_uploader("Subir archivo", type=["xlsx"])
            if data_file_xlsx is not None:
                df = pd.read_excel(data_file_xlsx)
                st.dataframe(df.head())
                ejecutar(df, op_algoritmo)

        elif op_archivo == "JSON":
            st.subheader("Seleccionar archivo JSON")
            data_file_json = st.file_uploader("Subir archivo", type=["json"])
            if data_file_json is not None:
                df = pd.read_json(data_file_json)
                st.dataframe(df.head())
                ejecutar(df, op_algoritmo)

    except Exception as e:
        st.write(e)

def ejecutar(df, op_algoritmo):

    try:

        opcionColumnas = df.keys()                    

        if op_algoritmo == "Regresion Lineal":
            campo1 = st.selectbox("CAMPO 1", opcionColumnas)
            campo2 = st.selectbox("CAMPO 2", opcionColumnas)  
            st.sidebar.write("Mostrar Grafica de dispercion:")
            if st.sidebar.button("DISPERSION"):                    
                grafica_dispersion(df, campo1, campo2)
            datoPredic =  st.sidebar.text_input("Ingresar dato a predecir (separado por coma):")
            st.sidebar.write("Mostrar datos:")
            if st.sidebar.button("REGRESION LINEAL"):
                if datoPredic != "":
                    arrayPredict = datoPredic.split(',')
                    regresion_lineal(df, campo1, campo2, arrayPredict)
                else:
                    st.error("Debe ingresar al menos un dato a predecir. Ej: 10,11,12")
        elif op_algoritmo == "Regresion Polinomial":
            campo1 = st.selectbox("CAMPO 1", opcionColumnas)
            campo2 = st.selectbox("CAMPO 2", opcionColumnas)  
            st.sidebar.write("Mostrar Grafica de dispercion:")
            if st.sidebar.button("DISPERSION"):                    
                grafica_dispersion(df, campo1, campo2)
            grado =  st.sidebar.text_input("Ingresar grado:")
            datoPredic =  st.sidebar.text_input("Ingresar dato a predecir:")
            st.sidebar.write("Mostrar datos:")
            if st.sidebar.button("REGRESION POLINOMIAL"):
                if datoPredic != "" and grado != "":
                    arrayPredict = datoPredic.split(',')
                    regresion_polinomial(df, campo1, campo2, grado, arrayPredict)
                else:
                    st.error("Debe ingresar un dato a predecir. Ej: 10 O grado de ecuacion. Ej: 2")
        elif op_algoritmo == "Clasificador Gaussiano":            
            atributos = st.multiselect("SELECCIONE ATRIBUTOS:", (opcionColumnas))
            clase = st.selectbox("SELECCIONE CLASE:", opcionColumnas)
            datoPredicG =  st.sidebar.text_input("Ingresar datos a predecir (separados por coma):")
            st.sidebar.write("Mostrar datos:")
            if st.sidebar.button("CLASIFICADOR GAUSSIANO"):
                if datoPredicG != "":
                    arrayPredict = datoPredicG.split(',')
                    gaussiano(df, atributos, clase, arrayPredict)
                else:
                    st.error("Debe ingresar datos a predecir. Ej: 10,3,2,1")
        elif op_algoritmo == "Arboles de decision":
            atributos = st.multiselect("SELECCIONE ATRIBUTOS:", (opcionColumnas))
            clase = st.selectbox("SELECCIONE CLASE:", opcionColumnas)
            if st.sidebar.button("ARBOLES DE DECISION"):
                arboles(df, atributos, clase)
        elif op_algoritmo == "Redes Neuronales":
            campo1 = st.selectbox("CAMPO 1", opcionColumnas)
            campo2 = st.selectbox("CAMPO 2", opcionColumnas)  
            datoPredic =  st.sidebar.text_input("Ingresar dato a predecir (separado por coma):")
            st.sidebar.write("Mostrar datos:")
            if st.sidebar.button("REDES NEURONALES"):
                if datoPredic != "":
                    arrayPredict = datoPredic.split(',')
                    redesNeuronales(df, campo1, campo2, arrayPredict)
                else:
                    st.error("Debe ingresar al menos un dato a predecir. Ej: 10")

    except Exception as e:
        st.write(e)

def grafica_dispersion(mi_df, mi_x, mi_y):

    try:

        fig = plt.figure()
        plt.xlabel(mi_x)
        plt.ylabel(mi_y)
        plt.scatter(mi_df[mi_x], mi_df[mi_y], color="green")
        st.pyplot(fig)

    except Exception as e:
        st.write(e)

def regresion_lineal(mi_df, mi_x, mi_y, arrayP):

    try:

        fig = plt.figure()
        regresion = linear_model.LinearRegression()
        x = mi_df[mi_x].values.reshape((-1,1))
        y = mi_df[mi_y]
        modelo = regresion.fit(x, mi_df[mi_y])

        #recorro, convierto y agrego datos, para que quede [[],[],[]]
        yPredic = []
        for pp in arrayP:
            np = []
            np.append(int(pp))
            yPredic.append(np)

        #Calculos
        intercepcionB = modelo.intercept_
        pendiente = modelo.coef_[0]
        ypredi = modelo.predict(x)
        r2 = r2_score(y, ypredi)
        err_cuadratico = mean_squared_error(y, ypredi)
        arrPredict = modelo.predict(yPredic)

        #Mostrando datos
        st.write("Ecuacion: ", "Y = (", pendiente, ")X + (", intercepcionB,")")
        st.write("R^2: ", r2)
        st.write("Error cuadratico: ", err_cuadratico)
        st.write("Prediccion: ")
        #Mostrando valores de prediccion
        for i in range(len(arrayP)):
            st.write(arrayP[i], " = ", arrPredict[i])
            
        #graficar puntos de prediccion    
        plt.scatter(yPredic, arrPredict, color="red")

        #graficar linea recta    
        plt.plot(x, ypredi, color="black")

        #graficar dispersion, con nombres de ejes
        plt.xlabel(mi_x)
        plt.ylabel(mi_y)
        plt.scatter(mi_df[mi_x], mi_df[mi_y], color="green")

        st.pyplot(fig)

    except Exception as e:
        st.write(e)

def regresion_polinomial(mi_df, mi_x, mi_y, grado, arrPredict):

    try:

        fig = plt.figure()

        regresion = linear_model.LinearRegression()
        pf = PolynomialFeatures(degree = int(grado))

        x = mi_df[mi_x].values.reshape((-1,1))
        y = mi_df[mi_y]   

        x_trans = pf.fit_transform(x)
        modelo = regresion.fit(x_trans, y)

        y_pred = regresion.predict(x_trans)

        x_new_min = int(arrPredict[0])
        x_new_max = int(arrPredict[0])
        x_new = np.linspace(x_new_min, x_new_max, 1)
        x_new = x_new[:, np.newaxis]

        x_trans = pf.fit_transform(x_new)    

        #Calculos
        y_prediccion = regresion.predict(x_trans)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        coeficientes = modelo.coef_
        intercepcion = modelo.intercept_
        #ecuacion
        text = "Y = " + str(intercepcion)
        for i in range(len(coeficientes)):
            text += " + (" + str(coeficientes[i])+")*X^"+str(i)

        #Aviso
        if len(arrPredict) > 1:
            st.warning("Ingreso mas de un valor a predecir, se tomara en cuenta solo el primero...")

        #Mostrando datos
        st.write("Ecuacion: ", text)
        st.write("R^2: ", r2)
        st.write("RMSE: ", rmse)
        st.write("Prediccion: ")
        st.write(arrPredict[0], " = ", y_prediccion[0])

        plt.scatter(x_new, y_prediccion, color="red")

        plt.scatter(x, y, color='green')
        plt.plot(x, y_pred, color='blue')
        st.pyplot(fig)

    except Exception as e:
        st.write(e)

def gaussiano(mi_df, atributos, clase, arrayPredict):

    try:

        clf = GaussianNB()
        le = preprocessing.LabelEncoder()
        #Creando array de arrays con los atributos
        arrArr_atributos = []
        for at in atributos:
            if type(mi_df[at][0]) == np.float64 or type(mi_df[at][0]) == np.int64 :
                newArr = mi_df[at]
                arrArr_atributos.append(newArr)
            else:
                newArr = le.fit_transform(mi_df[at])
                arrArr_atributos.append(newArr)

        x = list(zip(*arrArr_atributos))
        y = np.array(mi_df[clase])

        newArrayPredict = []
        for at in arrayPredict:
            newArrayPredict.append(int(at))
        
        clf.fit(x,y)
        mi_predic = clf.predict([newArrayPredict])

        st.write("PREDICCION: ", mi_predic)
        st.write("FEATURES:")
        st.write(x)
        st.write("CLASE:")
        st.write(y)

    except Exception as e:
        st.write(e)

def arboles(mi_df, atributos, clase):

    try:

        le = preprocessing.LabelEncoder()

        fig = plt.figure()

        arrArr_atributos = []
        for at in atributos:
            if type(mi_df[at][0]) == np.float64 or type(mi_df[at][0]) == np.int64 :
                newArr = mi_df[at]
                arrArr_atributos.append(newArr)
            else:
                newArr = le.fit_transform(mi_df[at])
                arrArr_atributos.append(newArr)

        x = list(zip(*arrArr_atributos))
        y = np.array(mi_df[clase])

        clf = DecisionTreeClassifier().fit(x,y)
        plot_tree(clf, filled=True)

        st.pyplot(fig)

        st.write("FEATURES:")
        st.write(x)
        st.write("CLASE:")
        st.write(y)

    except Exception as e:
        st.write(e)

def redesNeuronales(mi_df, mi_x, mi_y, arrayP):

    try:

        x = mi_df[mi_x]
        y = mi_df[mi_y]

        X = x[:,np.newaxis]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        mlr = MLPRegressor(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (3,3), random_state = 1)
        mlr.fit(X_train, y_train)

        #Aviso
        if len(arrayP) > 1:
            st.warning("Ingreso mas de un valor a predecir, se tomara en cuenta solo el primero...")

        scoree = mlr.score(X_train, y_train)
        st.write("Score: ", scoree)

        prediccion  = mlr.predict([[int(arrayP[0])]])
        st.write("Prediccion: ")
        st.write(arrayP[0]," = ",prediccion[0])
    
    except Exception as e:
        st.write(e)


if __name__ == '__main__':
    main()