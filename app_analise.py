import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from fpdf import FPDF
import time
from io import BytesIO


# Função para carregar e preparar os dados de imagem
def load_image_data(path, categories, img_size):
    data = []
    for category in categories:
        category_path = os.path.join(path, category)
        if not os.path.exists(category_path):
            st.error(f"O diretório {category_path} não foi encontrado.")
            return []
        for img in os.listdir(category_path):
            try:
                img_array = cv2.imread(os.path.join(category_path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                data.append([new_array, categories.index(category)])
            except Exception as e:
                pass
    return data


# Função para gerar o relatório em PDF
def generate_pdf_report(metrics_df, accuracy_fig, loss_fig):
    pdf = FPDF()
    pdf.add_page()

    # Adicionar título
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Métricas do Modelo", ln=True, align="C")

    # Adicionar tabela de métricas
    pdf.set_font("Arial", size=10)
    for i in range(len(metrics_df)):
        row = metrics_df.iloc[i]
        pdf.cell(200, 10, txt=str(row), ln=True)

    # Adicionar gráficos
    pdf.add_page()
    accuracy_fig_buffer = BytesIO()  # Usando BytesIO aqui
    accuracy_fig.savefig(accuracy_fig_buffer, format='png')
    accuracy_fig_buffer.seek(0)
    pdf.image(accuracy_fig_buffer, x=10, y=10, w=190)

    pdf.add_page()
    loss_fig_buffer = BytesIO()  # Usando BytesIO aqui
    loss_fig.savefig(loss_fig_buffer, format='png')
    loss_fig_buffer.seek(0)
    pdf.image(loss_fig_buffer, x=10, y=10, w=190)

    pdf_output = BytesIO()  # Usando BytesIO aqui
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return pdf_output


# Definindo os parâmetros e categorias do modelo
CATEGORIES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMG_SIZE = 200

# Variáveis globais
timer_start = 0
timer_running = False

# Título do aplicativo no Streamlit
st.title('Classificação de imagens usando CNN com Keras e CIFAR-10')
st.markdown("""
**Disciplina: Visão Computacional**  
**Professor:** Alex Cordeiro  
**Alunos:**   
    João Pedro 
    José Victor   
    Lucio Flavio  
    Néliton Vanderley  
    Wellington França  
""")
st.markdown("""
Este projeto foi desenvolvido com base em técnicas descritas no artigo "[Image Classification Using CNN with Keras & CIFAR-10](https://www.analyticsvidhya.com/blog/2021/01/image-classification-using-convolutional-neural-networks-a-step-by-step-guide/)",
 que fornece uma visão abrangente sobre a aplicação de redes neurais convolucionais na
  classificação de células sanguíneas, destacando a importância da automação na análise de 
  imagens médicas para melhorar a precisão e a eficiência dos diagnósticos laboratoriais.
""")
st.markdown("[Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)")

# Etapa 1: Upload do dataset
uploaded_file = st.file_uploader("Faça o upload de um arquivo zip com o dataset", type="zip")

if uploaded_file is not None:
    # Salva o arquivo zip carregado
    with open("uploaded_dataset.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Upload concluído!")

    # Extrai o arquivo zip para um diretório
    with zipfile.ZipFile("uploaded_dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("uploaded_dataset")
    st.success("Dataset extraído com sucesso!")

    path_test = "uploaded_dataset"

# Input para escolher o número de épocas
nb_epochs = st.number_input('Escolha o número de épocas', min_value=1, max_value=20, value=5)

# Botão para iniciar a análise e o treinamento do modelo
if st.button('Iniciar Análise e Treinamento'):
    timer_start = time.time()
    timer_running = True
    if path_test:
        # Etapa 2: Preparar conjunto de dados para treinamento
        st.write("Preparando conjunto de dados")
        st.write("STATUS: Processando...")
        path_test = path_test + "/dataset2-master/dataset2-master/images/TRAIN"
        training_data = load_image_data(path_test, CATEGORIES, IMG_SIZE)

        if not training_data:
            st.write("STATUS: ERRO")
            st.error("Erro ao carregar os dados. Verifique a estrutura do dataset.")
        else:
            # Etapa 3: Embaralhar o conjunto de dados
            random.shuffle(training_data)

            # Etapa 4: Atribuindo rótulos e recursos
            X = []
            y = []
            for features, label in training_data:
                X.append(features)
                y.append(label)

            X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            y = np.array(y)

            # Etapa 5: Normalizando X e convertendo rótulos em dados categóricos
            X = X.astype('float32') / 255.0
            y = tf.keras.utils.to_categorical(y, 4)

            # Etapa 6: Dividir X e Y para uso na CNN
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

            # Etapa 7: Definir, compilar e treinar o modelo CNN
            batch_size = 16
            nb_classes = 4

            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                       input_shape=(IMG_SIZE, IMG_SIZE, 3)),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(nb_classes, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            st.write("STATUS: Concluido!")

            # Treinamento do modelo
            st.write("Treinando o Modelo")
            st.write("STATUS: Em andamento...")
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1,
                                validation_data=(X_test, y_test))

            # Etapa 8: Precisão e pontuação do modelo
            score = model.evaluate(X_test, y_test, verbose=0)
            st.write(f"Test Score: {score[0]}")
            st.write(f"Test Accuracy: {score[1]}")

            # Plotar a precisão e a perda
            st.write("Visualização da Acurácia e Perda")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history['accuracy'], label='Train Accuracy')
            ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax[0].legend()
            ax[0].set_title('Accuracy')
            ax[1].plot(history.history['loss'], label='Train Loss')
            ax[1].plot(history.history['val_loss'], label='Validation Loss')
            ax[1].legend()
            ax[1].set_title('Loss')
            st.pyplot(fig)

            # Salvar os gráficos como imagens
            accuracy_fig = "accuracy_plot.png"
            loss_fig = "loss_plot.png"
            fig.savefig(accuracy_fig, bbox_inches='tight')
            fig.savefig(loss_fig, bbox_inches='tight')

            # Gerar dataframe com métricas de avaliação e histórico de treinamento
            metrics_dict = {
                "Metric": ["Test Score", "Test Accuracy"],
                "Value": [score[0], score[1]]
            }
            metrics_df = pd.DataFrame(metrics_dict)

            training_history_dict = {
                "Epoch": list(range(1, nb_epochs + 1)),
                "Train Accuracy": history.history['accuracy'],
                "Val Accuracy": history.history['val_accuracy'],
                "Train Loss": history.history['loss'],
                "Val Loss": history.history['val_loss']
            }
            training_history_df = pd.DataFrame(training_history_dict)
            csv_data = metrics_df.to_csv("model_metrics.csv", index=False).encode("utf-8")

            # Botões para download dos relatórios
            st.write("Gerar relatórios:")
            col1, col2 = st.columns(2)
            with col1:
                # Salva o dataframe como CSV temporário
                st.download_button(label="Download Relatório CSV", data=csv_data, file_name="model_metrics.csv")
            with col2:
                # Gera o relatório em PDF
                pdf_report_path = generate_pdf_report("model_metrics.csv", accuracy_fig, loss_fig)
                st.download_button(label="Download Relatório PDF", data=open(pdf_report_path, "rb").read(),
                                   file_name="model_report.pdf")

    else:
        st.error("Por favor, faça o upload do dataset antes de iniciar a análise e treinamento do modelo.")

# Botão para limpar todos os dados da análise
if st.button('Limpar Dados da Análise'):
    path_test = None
    timer_running = False
    st.success("Dados da análise limpos. Faça o upload de um novo dataset para iniciar uma nova análise.")

# Timer para indicar o tempo de análise
if timer_running:
    elapsed_time = (time.time() - timer_start)/60
    st.write(f"Tempo de análise decorrido: {elapsed_time:.2f} minutos.")

