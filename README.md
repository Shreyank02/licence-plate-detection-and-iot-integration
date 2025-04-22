# 🚗 License Plate Recognition with IoT Gate Automation 🔐

This project is a complete system for **automatic license plate recognition (ALPR)** integrated with an **IoT-based gate control system**. It uses **deep learning**, **computer vision**, **MySQL database verification**, and **Arduino servo control** to open a gate when a valid vehicle is detected.

---

## 📦 Project Overview

This system is designed to automate entry management for parking lots, residential societies, or restricted areas. It uses a camera to capture a car’s license plate, detects and recognizes the plate number using a trained CNN, and then checks the number against a MySQL database.

If the license plate is authorized, a signal is sent to an Arduino via serial communication, which then opens the gate by rotating a servo motor.

---

## ✨ Features

- ✅ License plate detection from image input
- 🔍 Character segmentation and classification using a CNN model
- 🧠 Uses TensorFlow/Keras for OCR
- 🗃 MySQL integration for plate verification
- 🧾 Modular code with separate files for detection, DB check, and hardware interface
- 🔌 Communicates with Arduino using pyserial
- 🔁 Controls a servo motor to simulate gate open/close
- 💻 Easily extendable for real-time video or cloud database integration

---

## 💡 Use Cases

- Smart parking systems
- Gated communities
- Office building security
- Vehicle tracking systems
- IoT-based automation projects

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend**  | *(Optional)* Streamlit *(future enhancement)* |
| **Processing** | Python, OpenCV, TensorFlow, Keras |
| **Database** | MySQL |
| **Hardware**  | Arduino Uno + Servo Motor |
| **Communication** | Serial via `pyserial` |

---

## 🗂️ Folder Structure


