import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tifffile
import numpy as np

class MainApplication(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Tiff Stack Creator")
        self.geometry("300x200")

        self.input_files = []
        self.output_file = tk.StringVar()
        
        self.input_files_label = tk.Label(self, text="Input Files:")
        self.input_files_label.pack()
        
        self.input_files_listbox = tk.Listbox(self)
        self.input_files_listbox.pack()
        
        self.add_files_button = tk.Button(self, text="Add Files", command=self.add_files)
        self.add_files_button.pack()
        
        self.remove_files_button = tk.Button(self, text="Remove Files", command=self.remove_files)
        self.remove_files_button.pack()
        
        self.move_up_button = tk.Button(self, text="Move Up", command=self.move_up)
        self.move_up_button.pack()
        
        self.move_down_button = tk.Button(self, text="Move Down", command=self.move_down)
        self.move_down_button.pack()
        
        self.output_file_label = tk.Label(self, text="Output File:")
        self.output_file_label.pack()
        
        self.output_file_entry = tk.Entry(self, textvariable=self.output_file)
        self.output_file_entry.pack()
        
        self.choose_output_button = tk.Button(self, text="Choose Output", command=self.choose_output)
        self.choose_output_button.pack()
        
        self.create_stack_button = tk.Button(self, text="Create Stack", command=self.create_stack)
        self.create_stack_button.pack()

    def add_files(self):
        files = filedialog.askopenfilenames(title="Choose Input Files", filetypes=[("TIFF Files", "*.tiff;*.tif")])
        for file in files:
            self.input_files.append(file)
            self.input_files_listbox.insert(tk.END, file)

    def remove_files(self):
        selected_indices = self.input_files_listbox.curselection()
        for index in reversed(selected_indices):
            self.input_files_listbox.delete(index)
            self.input_files.pop(index)
            
    def move_up(self):
        selected_index = self.input_files_listbox.curselection()
        if selected_index and selected_index[0] > 0:
            selected_file = self.input_files.pop(selected_index[0])
            self.input_files.insert(selected_index[0] - 1, selected_file)
            self.input_files_listbox.delete(selected_index)
            self.input_files_listbox.insert(selected_index[0] - 1, selected_file)
            self.input_files_listbox.selection_clear(0, tk.END)
            self.input_files_listbox.activate(selected_index[0] - 1)
            self.input_files_listbox.selection_set(selected_index[0] - 1, last=None)
            
    def move_down(self):
        selected_index = self.input_files_listbox.curselection()
        if selected_index and selected_index[0] < self.input_files_listbox.size() - 1:
            selected_file = self.input_files.pop(selected_index[0])
            self.input_files.insert(selected_index[0] + 1, selected_file)
            self.input_files_listbox.delete(selected_index)
            self.input_files_listbox.insert(selected_index[0] + 1, selected_file)
            self.input_files_listbox.selection_clear(0, tk.END)
            self.input_files_listbox.activate(selected_index[0] + 1)
            self.input_files_listbox.selection_set(selected_index[0] + 1, last=None)
    
    def choose_output(self):
        file = filedialog.asksaveasfilename(title="Choose Output File", filetypes=[("TIFF Files", "*.tiff;*.tif")], defaultextension=".tiff")
        self.output_file.set(file)
        
    def create_stack(self):
        if not self.input_files:
            messagebox.showerror("Error", "No input files selected")
            return
        if not self.output_file.get():
            messagebox.showerror("Error", "No output file selected")
            return
        try:
            images = []
            depths = []
            i = 0
            for file in self.input_files:
                image = tifffile.imread(file)
                images.append(image)
                depths.append(image.shape[0])
            stack_depth = sum(depths)
            stack = np.zeros((stack_depth, image.shape[1], image.shape[2]), dtype=images[0].dtype)
            for file in self.input_files:
                for j in range(tifffile.imread(file).shape[0]):
                    stack[i][:][:] = tifffile.imread(file)[j]
                    i += 1
            tifffile.imsave(self.output_file.get(), stack)
            messagebox.showinfo("Success", "Stack created successfully")
        except Exception as e:
            messagebox.showerror("Error", "Failed to create stack: " + str(e))

app = MainApplication()
app.mainloop()