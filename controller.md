# 🎮 **Joystick Controls Summary**

This node maps joystick inputs to **end-effector Cartesian velocities** and **gripper actions** for Sawyer via RelaxedIK.

#### **Left Stick – Linear Motion**

| Control                                   | Action                                          |
| ----------------------------------------- | ----------------------------------------------- |
| **Left Stick Up/Down**                    | Move end-effector **forward/backward** (x-axis) |
| **Left Stick Left/Right**                 | Move **left/right** (y-axis)                    |
| **Right Trigger Vertical Axis (axes[4])** | Move **up/down** (z-axis)                       |

#### **Right Stick – Orientation**

| Control                    | Action           |
| -------------------------- | ---------------- |
| **Right Stick Left/Right** | Rotate **roll**  |
| **Right Stick Up/Down**    | Rotate **pitch** |

#### **Hold LT (Left Trigger) → Yaw Mode**

When **LT is pressed**, the right stick switches to yaw control:

| Control                              | Action                      |
| ------------------------------------ | --------------------------- |
| **Right Stick Left/Right (with LT)** | Rotate **yaw**              |
| **Pitch/Roll disabled**              | Ensures precise yaw control |

#### **Gripper Control**

| Button           | Action            |
| ---------------- | ----------------- |
| **A (button 0)** | **Close** gripper |
| **B (button 1)** | **Open** gripper  |

#### **Other Features**

* All axes use **deadzone filtering** to remove drift.
* **Cubic scaling** provides fine control near the center.
* Velocities are **smoothed (α = 0.2)** for stable Cartesian motion.
* Commands publish at **30 Hz** (`TimerCallback`).

---


# ⌨️ **Keyboard Controls Summary**

This node maps **keyboard keys to end-effector Cartesian velocities** and **gripper actions** for Sawyer via RelaxedIK.

### **Linear Motion (Position Control)**

| Key   | Action             |
| ----- | ------------------ |
| **W** | Move Forward (+X)  |
| **S** | Move Backward (–X) |
| **A** | Move Left (+Y)     |
| **D** | Move Right (–Y)    |
| **R** | Move Up (+Z)       |
| **F** | Move Down (–Z)     |

---

### **Angular Motion (Orientation Control)**

| Key   | Action  |
| ----- | ------- |
| **1** | Roll +  |
| **2** | Roll –  |
| **3** | Pitch + |
| **4** | Pitch – |
| **5** | Yaw +   |
| **6** | Yaw –   |

---

### **Gripper Control**

| Key   | Action        |
| ----- | ------------- |
| **O** | Open Gripper  |
| **P** | Close Gripper |

---

### **Stride Adjustments (Movement Step Size)**

| Key            | Action                            |
| -------------- | --------------------------------- |
| **. (Period)** | Increase position stride by +0.01 |
| **, (Comma)**  | Decrease position stride by –0.01 |

---

### **Other**

| Key   | Action                             |
| ----- | ---------------------------------- |
| **C** | Quit teleoperation & shutdown node |

---

### **Additional Features**

* Smooth velocity blending using **α = 0.2**.
* Publishes EE velocity goals at **30 Hz**.
* Cubic scaling + deadzone filtering for stable control.
* Works with or without an electric gripper.


