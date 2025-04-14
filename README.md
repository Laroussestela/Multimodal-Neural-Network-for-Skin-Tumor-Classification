# 🧠 Multimodal Neural Network for Tumor Classification

Este repositorio contiene una red neuronal **multimodal** diseñada para clasificar tumores combinando **imágenes médicas** y **datos estructurados** del paciente como edad, sexo y localización del tumor.

## 🚀 Descripción del Proyecto

La arquitectura del modelo está compuesta por dos ramas principales:

- 🖼️ **Procesamiento de Imágenes**: Convolutional Neural Network (CNN)
- 📊 **Procesamiento de Datos Tabulares**: Información estructurada como:
  - Edad del paciente
  - Sexo
  - Localización anatómica del tumor

Ambas salidas se fusionan para generar una representación conjunta, que alimenta a una capa final de clasificación.

## 🧬 Dataset

El conjunto de datos está compuesto por:

- Imágenes de lesiones (RGB, tamaño 64x64 píxeles)
- Datos estructurados asociados a cada imagen (edad, sexo, localización)
- Etiquetas multiclase codificadas one-hot (7 clases posibles)

> 🔒 Por motivos de privacidad, los datos no se incluyen en este repositorio. Si deseas usar el código, asegúrate de tener acceso a un dataset similar o de adaptar el pipeline de carga de datos.

## 🏗️ Arquitectura del Modelo

![image](https://github.com/user-attachments/assets/1819f57a-cdb3-4873-bbee-a2080742ce75)

## 📋 Métricas Comparativas

| Modelo                     | Loss   | Accuracy | Precision | Recall | Specificidad | F1-Score |
|---------------------------|--------|----------|-----------|--------|---------------|----------|
| CNN (solo imagen)         | 0.7561 | 0.8720   | 0.8729    | 0.8705 | 0.9789        | 0.8717   |
| Red Neuronal Multimodal   | 0.5292 | 0.9149   | 0.9184    | 0.9120 | 0.9832        | 0.9151   |

🧠 **Conclusión**:  
La red neuronal multimodal **supera claramente** a la CNN tradicional, demostrando que **integrar información clínica adicional mejora significativamente el rendimiento del modelo**.

---

## 📉 Matriz de Confusión

A continuación, se muestra la **matriz de confusión** de la Red Neuronal Multimodal en el conjunto de test, para las 7 clases posibles del problema de clasificación:

![image](https://github.com/user-attachments/assets/ba386eb3-a59b-413c-b810-5dd6c4249e77)



