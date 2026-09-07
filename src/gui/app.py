import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from src.processing.canny import detect_canny_edges
from src.processing.marr_hildreth import detect_marr_hildreth_edges
from src.processing.sharpening import sharpen_image


# --------------------------------------------------
# Appearance
# --------------------------------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImageSharpeningApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("ImageSharpeningTool")
        self.geometry("1400x850")
        self.minsize(1150, 700)

        self.image_path = None
        self.original_image = None
        self.edge_image = None
        self.result_image = None

        self.method = tk.StringVar(value="Canny")
        self.strength = tk.DoubleVar(value=0.5)

        self.create_layout()

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    def create_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # =========================
        # Sidebar
        # =========================
        self.sidebar = ctk.CTkFrame(
            self,
            width=300,
            corner_radius=0,
            fg_color=("#E9EEF5", "#111827")
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Brand
        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=28, pady=(30, 8))

        ctk.CTkLabel(
            brand,
            text="Image",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(anchor="w")

        ctk.CTkLabel(
            brand,
            text="Sharpening Tool",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(anchor="w")

        ctk.CTkLabel(
            self.sidebar,
            text="Edge-Based Image Enhancement",
            text_color=("#64748B", "#94A3B8"),
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=28, pady=(0, 28))

        # Upload
        self.upload_button = ctk.CTkButton(
            self.sidebar,
            text="📂  Upload Image",
            height=48,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.upload_image
        )
        self.upload_button.pack(fill="x", padx=24, pady=(0, 22))

        # Section helper
        ctk.CTkLabel(
            self.sidebar,
            text="PROCESSING METHOD",
            text_color=("#475569", "#94A3B8"),
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=24, pady=(4, 8))

        self.canny_radio = ctk.CTkRadioButton(
            self.sidebar,
            text="Canny Edge Detection",
            variable=self.method,
            value="Canny"
        )
        self.canny_radio.pack(anchor="w", padx=30, pady=6)

        self.marr_radio = ctk.CTkRadioButton(
            self.sidebar,
            text="Marr-Hildreth (LoG)",
            variable=self.method,
            value="Marr-Hildreth"
        )
        self.marr_radio.pack(anchor="w", padx=30, pady=6)

        # Strength
        ctk.CTkLabel(
            self.sidebar,
            text="SHARPENING STRENGTH",
            text_color=("#475569", "#94A3B8"),
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=24, pady=(26, 6))

        strength_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        strength_row.pack(fill="x", padx=24)

        self.strength_value = ctk.CTkLabel(
            strength_row,
            text="0.5",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.strength_value.pack(side="right")

        self.strength_slider = ctk.CTkSlider(
            self.sidebar,
            from_=0.1,
            to=2.0,
            number_of_steps=19,
            variable=self.strength,
            command=self.update_strength
        )
        self.strength_slider.pack(fill="x", padx=24, pady=(2, 4))

        # Action buttons
        self.sharpen_button = ctk.CTkButton(
            self.sidebar,
            text="✨  Sharpen Image",
            height=48,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.process_image
        )
        self.sharpen_button.pack(fill="x", padx=24, pady=(28, 9))

        self.save_button = ctk.CTkButton(
            self.sidebar,
            text="💾  Save Result",
            height=43,
            corner_radius=10,
            fg_color="transparent",
            border_width=1,
            command=self.save_result
        )
        self.save_button.pack(fill="x", padx=24, pady=5)

        self.reset_button = ctk.CTkButton(
            self.sidebar,
            text="↻  Reset",
            height=40,
            corner_radius=10,
            fg_color="transparent",
            hover_color=("#DCE4EF", "#1F2937"),
            command=self.reset_app
        )
        self.reset_button.pack(fill="x", padx=24, pady=5)

        # Status at bottom
        status_box = ctk.CTkFrame(
            self.sidebar,
            corner_radius=10,
            fg_color=("#DCE4EF", "#1B2433")
        )
        status_box.pack(side="bottom", fill="x", padx=20, pady=22)

        self.status_label = ctk.CTkLabel(
            status_box,
            text="●  Ready",
            text_color="#22C55E",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.status_label.pack(anchor="w", padx=14, pady=10)

        # =========================
        # Main content
        # =========================
        self.content = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color=("#F4F7FB", "#0B1120")
        )
        self.content.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=0,
            pady=0
        )

        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self.content, fg_color="transparent")
        header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=30,
            pady=(28, 18)
        )
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Image Preview",
            font=ctk.CTkFont(size=30, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        self.image_info = ctk.CTkLabel(
            header,
            text="No image selected",
            text_color=("#64748B", "#94A3B8"),
            font=ctk.CTkFont(size=12)
        )
        self.image_info.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.method_badge = ctk.CTkLabel(
            header,
            text="CANNY",
            width=90,
            height=30,
            corner_radius=15,
            fg_color=("#DBEAFE", "#172554"),
            text_color=("#1D4ED8", "#60A5FA"),
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.method_badge.grid(row=0, column=1, rowspan=2, padx=10)

        self.method.trace_add("write", self.update_method_badge)

        # Preview cards
        self.preview_frame = ctk.CTkFrame(
            self.content,
            fg_color="transparent"
        )
        self.preview_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=24,
            pady=(0, 24)
        )

        for i in range(3):
            self.preview_frame.grid_columnconfigure(i, weight=1, uniform="cards")
        self.preview_frame.grid_rowconfigure(0, weight=1)

        self.original_card = self.create_image_card(
            self.preview_frame, "Original Image", "SOURCE", 0
        )
        self.edge_card = self.create_image_card(
            self.preview_frame, "Edge Image", "EDGE MAP", 1
        )
        self.result_card = self.create_image_card(
            self.preview_frame, "Sharpened Image", "RESULT", 2
        )

    # --------------------------------------------------
    # Image card
    # --------------------------------------------------

    def create_image_card(self, parent, title, tag, column):
        card = ctk.CTkFrame(
            parent,
            corner_radius=16,
            fg_color=("#FFFFFF", "#111827"),
            border_width=1,
            border_color=("#E2E8F0", "#243047")
        )
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=7
        )

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=18, pady=(16, 8))

        ctk.CTkLabel(
            top,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left")

        ctk.CTkLabel(
            top,
            text=tag,
            text_color=("#64748B", "#64748B"),
            font=ctk.CTkFont(size=9, weight="bold")
        ).pack(side="right")

        image_area = ctk.CTkFrame(
            card,
            corner_radius=12,
            fg_color=("#F1F5F9", "#0F172A")
        )
        image_area.pack(
            expand=True,
            fill="both",
            padx=15,
            pady=(4, 15)
        )

        image_label = ctk.CTkLabel(
            image_area,
            text="No Image\n\nUpload an image to begin",
            text_color=("#64748B", "#64748B"),
            font=ctk.CTkFont(size=13)
        )
        image_label.pack(expand=True, fill="both", padx=10, pady=10)

        return image_label

    # --------------------------------------------------
    # Upload
    # --------------------------------------------------

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        if image is None:
            messagebox.showerror(
                "Error",
                "Could not open the selected image."
            )
            return

        self.image_path = file_path
        self.original_image = image
        self.edge_image = None
        self.result_image = None

        height, width = image.shape[:2]
        filename = file_path.replace("\\", "/").split("/")[-1]

        self.image_info.configure(
            text=f"{filename}   •   {width} × {height}px"
        )

        self.display_image(image, self.original_card)
        self.clear_preview(self.edge_card, "Edge Image")
        self.clear_preview(self.result_card, "Sharpened Image")

        self.status_label.configure(
            text="●  Image loaded",
            text_color="#22C55E"
        )

    # --------------------------------------------------
    # Strength
    # --------------------------------------------------

    def update_strength(self, value):
        value = round(float(value), 1)
        self.strength_value.configure(text=str(value))

    # --------------------------------------------------
    # Processing
    # --------------------------------------------------

    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning(
                "No Image",
                "Please upload an image first."
            )
            return

        self.status_label.configure(
            text="●  Processing...",
            text_color="#F59E0B"
        )
        self.sharpen_button.configure(state="disabled")
        self.update()

        try:
            if self.method.get() == "Canny":
                self.edge_image = detect_canny_edges(
                    self.original_image
                )
            else:
                self.edge_image = detect_marr_hildreth_edges(
                    self.original_image
                )

            self.result_image = sharpen_image(
                self.original_image,
                self.edge_image,
                self.strength.get()
            )

            self.display_image(self.edge_image, self.edge_card)
            self.display_image(self.result_image, self.result_card)

            self.status_label.configure(
                text="●  Processing complete",
                text_color="#22C55E"
            )

        except Exception as error:
            self.status_label.configure(
                text="●  Processing failed",
                text_color="#EF4444"
            )
            messagebox.showerror(
                "Processing Error",
                f"Could not process the image:\n\n{error}"
            )

        finally:
            self.sharpen_button.configure(state="normal")

    # --------------------------------------------------
    # Display image
    # --------------------------------------------------

    def display_image(self, image, label):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)

        # Keep the image inside the card without distortion.
        max_width = 430
        max_height = 560

        pil_image.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(pil_image)

        label.configure(
            image=photo,
            text=""
        )
        label.image = photo

    def clear_preview(self, label, title):
        label.configure(
            image=None,
            text=f"No {title.replace(' Image', '').lower()} yet"
        )
        label.image = None

    # --------------------------------------------------
    # Method badge
    # --------------------------------------------------

    def update_method_badge(self, *_):
        method = self.method.get()

        if method == "Canny":
            self.method_badge.configure(text="CANNY")
        else:
            self.method_badge.configure(text="MARR-HILDRETH")

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset_app(self):
        self.image_path = None
        self.original_image = None
        self.edge_image = None
        self.result_image = None

        self.method.set("Canny")
        self.strength.set(0.5)
        self.strength_value.configure(text="0.5")

        self.clear_preview(self.original_card, "Original Image")
        self.clear_preview(self.edge_card, "Edge Image")
        self.clear_preview(self.result_card, "Sharpened Image")

        self.image_info.configure(text="No image selected")

        self.status_label.configure(
            text="●  Ready",
            text_color="#22C55E"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    def save_result(self):
        if self.result_image is None:
            messagebox.showwarning(
                "No Result",
                "Please sharpen an image first."
            )
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Sharpened Image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG Image", "*.jpg"),
                ("PNG Image", "*.png")
            ]
        )

        if not file_path:
            return

        success = cv2.imwrite(file_path, self.result_image)

        if success:
            self.status_label.configure(
                text="●  Result saved",
                text_color="#22C55E"
            )
            messagebox.showinfo(
                "Saved",
                "Sharpened image saved successfully."
            )
        else:
            messagebox.showerror(
                "Save Error",
                "Could not save the result image."
            )


# --------------------------------------------------
# Start Application
# --------------------------------------------------

if __name__ == "__main__":
    app = ImageSharpeningApp()
    app.mainloop()
