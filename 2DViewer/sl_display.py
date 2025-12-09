#!/usr/bin/env python3
"""
Interactive MRCS stack viewer
"""

import sys, numpy as np, mrcfile
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------
# Image tile
# ---------------------------------------------------------------------
class ImageItem(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, array: np.ndarray, index: int, parent_view, entropy=None):
        super().__init__()
        self.raw = array.astype(np.float32)
        self.index = index
        self.entropy = entropy  # store entropy for sorting
        self.view = parent_view  # back-reference to MRCViewer for brightness/contrast
        self.selected = False
        self.border = QtWidgets.QGraphicsRectItem(self)
        self.border.setPen(QtGui.QPen(QtCore.Qt.red, 2))
        self.border.setVisible(False)
        # Add label for ID in lower left
        #self.id_label = QtWidgets.QGraphicsSimpleTextItem(str(self.index), self)
        self.id_label = QtWidgets.QGraphicsSimpleTextItem(str(self.index + 1), self)

        self.id_label.setBrush(QtGui.QBrush(QtGui.QColor(0, 200, 0)))  # green
        font = QtGui.QFont()
        font.setPointSize(12)
        self.id_label.setFont(font)
        self.updatePixmap()

    def updatePixmap(self):
        arr = self.raw
        amin, amax = float(arr.min()), float(arr.max())
        if amax > amin:
            norm = (arr - amin) / (amax - amin) * 255.0  # 0-255 float
        else:
            norm = np.zeros_like(arr, dtype=np.float32)

        img = (norm - 128.0) * self.view.contrast + 128.0 + self.view.brightness
        img8 = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img8.shape
        qimg = QtGui.QImage(img8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self.border.setRect(0, 0, w, h)
        self.border.setVisible(self.selected)
        self.id_label.setPos(3, h - self.id_label.boundingRect().height() - 2)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton and not e.modifiers():
            self.selected = not self.selected
            self.border.setVisible(self.selected)
            e.accept()
        else:
            super().mousePressEvent(e)

# ---------------------------------------------------------------------
# Brightness / Contrast dialog
# ---------------------------------------------------------------------
class BCDialog(QtWidgets.QWidget):
    def __init__(self, viewer):
        super().__init__(viewer, QtCore.Qt.Window)
        self.viewer = viewer
        self.setWindowTitle("Brightness / Contrast")
        lay = QtWidgets.QVBoxLayout(self)

        self.b = QtWidgets.QSlider(QtCore.Qt.Horizontal, tracking=False)
        self.b.setRange(-128, 128)
        lay.addWidget(QtWidgets.QLabel("Brightness (offset)"))
        lay.addWidget(self.b)

        self.c = QtWidgets.QSlider(QtCore.Qt.Horizontal, tracking=False)
        self.c.setRange(10, 500)
        lay.addWidget(QtWidgets.QLabel("Contrast (×)"))
        lay.addWidget(self.c)

        # live preview while dragging → use sliderMoved
        self.b.sliderMoved.connect(self.apply)
        self.c.sliderMoved.connect(self.apply)
        # commit when released
        self.b.valueChanged.connect(self.apply)
        self.c.valueChanged.connect(self.apply)

    def showEvent(self, e):
        self.b.setValue(int(self.viewer.brightness))
        self.c.setValue(int(self.viewer.contrast * 100))
        super().showEvent(e)

    def apply(self, _):
        self.viewer.updateContrast(float(self.b.value()), float(self.c.value())/100)

# ---------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------
class Viewer(QtWidgets.QGraphicsView):
    def __init__(self, path, cols=5):
        super().__init__()
        self.setWindowTitle("dxdisplay")  ### Set window title here
        self.brightness, self.contrast = 0.0, 1.0
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.grabGesture(QtCore.Qt.PinchGesture)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.cols = cols  # store number of columns for reuse


        with mrcfile.open(path, permissive=True) as mrc:
            stack = mrc.data.astype(np.float32)

        # --- Calculate entropy for each image ---
        self.entropies = []
        for arr in stack:
            hist, _ = np.histogram(arr, bins=64, range=(arr.min(), arr.max()) if arr.max() > arr.min() else (0,1))
            e = scipy_entropy(hist) if np.any(hist > 0) else 0.0
            if e <= 0:
                e = 99999
            self.entropies.append(e)

        # --- Store image items (with entropy) ---
        self.tiles = []
        h, w = stack[0].shape
        for i, img in enumerate(stack):
            tile = ImageItem(img, i, self, self.entropies[i])
            r, c = divmod(i, cols)
            tile.setPos(c*(w+8), r*(h+8))
            self.scene.addItem(tile)
            self.tiles.append(tile)

        self.dialog = BCDialog(self)

        # actions / context menu
        save = QtWidgets.QAction("Save Selection", self, shortcut="Ctrl+S", triggered=self.save)
        save_list = QtWidgets.QAction("Save Selection as List", self, triggered=self.save_list)  ### NEW
        load_list = QtWidgets.QAction("Load Selection", self, triggered=self.load_list)  # <-- new line
        sort_entropy = QtWidgets.QAction("Sort by Entropy", self, triggered=self.sort_by_entropy)  ### NEW
        bc   = QtWidgets.QAction("Brightness / Contrast", self, shortcut="B", triggered=self.dialog.show)
        #self.addActions([save, save_list, sort_entropy, bc])
        self.addActions([save, save_list, load_list, sort_entropy, bc])
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

    # -----------------------------------------------------------------
    def wheelEvent(self, e):
        if e.modifiers() & QtCore.Qt.ControlModifier:
            s = 1.001 ** e.angleDelta().y()
            self.scale(s, s)
        else:
            super().wheelEvent(e)

    def event(self, e):
        if e.type() == QtCore.QEvent.Gesture:
            p = e.gesture(QtCore.Qt.PinchGesture)
            if p:
                self.scale(p.scaleFactor(), p.scaleFactor())
                return True
        return super().event(e)

    # -----------------------------------------------------------------
    def updateContrast(self, b, c):
        self.brightness, self.contrast = b, c
        for t in self.tiles:
            t.updatePixmap()

    def save(self):
        sel = [t.raw for t in self.tiles if t.selected]
        if not sel:
            QtWidgets.QMessageBox.information(self, "Save", "Nothing selected")
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save MRCS", filter="MRC (*.mrc *.mrcs)")
        if not fn:
            return
        with mrcfile.new(fn, overwrite=True) as m: m.set_data(np.stack(sel).astype(np.float32))
        #QtWidgets.QMessageBox.information(self, "Save", f"Saved {len(sel)} imgs → {fn}")

    def save_list(self):  ### NEW
        #sel_ids = [str(t.index) for t in self.tiles if t.selected]
        sel_ids = [str(t.index + 1) for t in self.tiles if t.selected]

        if not sel_ids:
            QtWidgets.QMessageBox.information(self, "Save List", "Nothing selected")
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save List", filter="Text files (*.txt)")
        if not fn:
            return
        with open(fn, "w") as f:
            for idx in sel_ids:
                f.write(idx + "\n")
        #QtWidgets.QMessageBox.information(self, "Save List", f"Saved {len(sel_ids)} indices to {fn}")


    def load_list(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Selection", filter="Text files (*.txt)")
        if not fn:
            return
        with open(fn) as f:
            ids = set(int(line.strip()) for line in f if line.strip().isdigit())
        count = 0
        for t in self.tiles:
            #t.selected = t.border.setVisible(t.index in ids)
            t.selected = t.border.setVisible((t.index + 1) in ids)

            if t.selected:
                count += 1
        #QtWidgets.QMessageBox.information(self, "Load Selection", f"{count} images selected.")


    def sort_by_entropy(self):  ### NEW
        # Sort tiles by entropy (lowest first, high/zero entropy last)
        # Keep the original green label (tile.index)
        sorted_tiles = sorted(self.tiles, key=lambda t: t.entropy)
        h = sorted_tiles[0].raw.shape[0]
        w = sorted_tiles[0].raw.shape[1]
        #cols = int(np.sqrt(len(sorted_tiles))) if len(sorted_tiles) > 1 else 1
        cols = self.cols

        if hasattr(self, 'scene'):
            for i, tile in enumerate(sorted_tiles):
                r, c = divmod(i, cols)
                tile.setPos(c*(w+8), r*(h+8))
        self.scene.update()
        # Optionally: de-select all tiles (comment out if not desired)
        # for t in self.tiles:
        #     t.selected = False
        #     t.border.setVisible(False)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("mrcs")
    a.add_argument("--cols", type=int, default=5)
    args = a.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    Viewer(args.mrcs, cols=args.cols).show()
    sys.exit(app.exec_())

