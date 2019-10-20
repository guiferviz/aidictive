"""Utility that allows you to paint in notebooks.

This is very useful to generate your own training or
test data. To see the algorithms working in your own
data is more satisfying than anything else!
"""


import uuid

import numpy as np


_html_code = """
    <div width="300">
        <h1>[[[title]]]</h1>
        <canvas id="canvas_[[[id]]]"
                width="[[[canvas_w]]]"
                height="[[[canvas_h]]]">
        </canvas>
        <div class="button-container">
            <button id="clear_btn_[[[id]]]" class="blue-btn">
                <i class="fa fa-eraser" aria-hidden="true"></i> Clear
            </button>
            <button id="add_btn_[[[id]]]" class="green-btn">
                <i class="fa fa-plus" aria-hidden="true"></i> Add
            </button>
        </div>
        <div id="gallery_[[[id]]]" class="gallery"></div>
    </div>

"""

_css_code = """
    <style>

        canvas {
            border: solid 1px;
        }

        .button-container {
            padding: 10px;
        }

        .button-container > button {
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
        }

        .gallery > img {
            display: inline-block;
            vertical-align: unset;
        }

        .green-btn {background-color: #4CAF50;}
        .blue-btn {background-color: #008CBA;}
        .red-btn {background-color: #f44336;}
        .white-btn {background-color: #e7e7e7; color: black !important;}
        .black-btn {background-color: #555555;}

    </style>
"""

_js_code = """
    <script src="https://code.jquery.com/jquery-3.1.0.min.js"></script>
    <script src="https://d3js.org/d3.v4.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/1.6.3/fabric.min.js"></script>
    <script>
        function whenAvailable(name, callback) {
            var interval = 10; // ms
            window.setTimeout(function() {
                if (window[name]) {
                    callback(window[name]);
                } else {
                    window.setTimeout(arguments.callee, interval);
                }
            }, interval);
        }

        function createCanvas(w, h){
            var canvas = document.createElement("canvas");
            canvas.width = w;
            canvas.height = h;

            return canvas;
        }

        function getImgFromCanvas(canvas)
        {
            var p = new Promise(resolve => {
                var img = new Image();
                img.onload = function () {
                    resolve(img)
                };
                img.src = canvas.toDataURL("png");
            });

            return p;
        }

        function resize(img, w, h)
        {
            var canvas = createCanvas(w, h);
            var ctx = canvas.getContext("2d");

            ctx.drawImage(img, 0, 0, w, h);

            return getImgFromCanvas(canvas);
        }

        function getBinaryMatrix(img, w, h)
        {
            var canvasAux = createCanvas(w, h);
            var ctxAux = canvasAux.getContext("2d");
            ctxAux.drawImage(img, 0, 0, w, h);

            var data = ctxAux.getImageData(0, 0, w, h).data;
            var x = new Array(h);
            for (var i = 0; i < h; i++) {
                x[i] = new Array(w);
            }

            for (var i = 0; i < data.length; i += 4) {
                var r = Math.floor(i / (4 * h))
                var c = (i / 4) % w
                var dark = (data[i] + data[i+1] + data[i+2]) / 3 <= 128
                var transparent = data[i + 3] <= 128
                x[r][c] = transparent || !dark ? 0 : 1;
            }

            return x;
        }

        function sendToPy(matrix) {
            var matrix_str = JSON.stringify(matrix);
            var kernel = IPython.notebook.kernel;
            command = "NotebookPainter._instances['[[[id]]]']._data.append(" + matrix_str + ")";
            console.log(command);
            kernel.execute(command);
        }

        whenAvailable("fabric", function () {
            var data = [];
            var canvas = new fabric.Canvas('canvas_[[[id]]]');
            canvas.isDrawingMode = true;
            canvas.freeDrawingBrush.width = [[[brush_width]]];
            canvas.freeDrawingBrush.color = "[[[brush_color]]]";
            //canvas.backgroundColor = "white";

            $("#clear_btn_[[[id]]]").click(function () { canvas.clear() })
            $("#add_btn_[[[id]]]").click(async function () {
                var w = [[[out_w]]];
                var h = [[[out_h]]];

                var img = await getImgFromCanvas(canvas);
                var resizedImg = await resize(img, w, h);
                $("#gallery_[[[id]]]").append(resizedImg);

                var matrix = getBinaryMatrix(img, w, h);
                sendToPy(matrix);
            });
        })
    </script>
"""


def apply_template(template, **kwargs):
    """Substitute all the variables in the template using kwargs.

    This "template engine" is oriented to programming codes
    templates. Instead of using `{variable_name}`, as
    Python's `str.format` function does, this function uses
    `[[[variable_name]]]`. Using this new way we avoid
    escaping the curly brackets that are widely used in
    languages like Javascript or CSS.

    Arguments:
        template (str): String with the template.
        **kwargs: Dict with pairs of variable keys
            and values.
    """
    # Escape curly brackets.
    template = template\
        .replace("{", "{{")\
        .replace("}", "}}")
    # Replace our selected mark by curly brackets.
    template = template\
        .replace("[[[", "{")\
        .replace("]]]", "}")
    # Format using classic Python format.
    return template.format(**kwargs)


class NotebookPainter(object):
    """Generate an HTML canvas you can draw on.

    Any object of this class returned by a cell of a
    Jupyter Notebook will show a HTML canvas in which
    you can draw. You can also save your draws and
    obtain a NumPy array with the image data.
    This component is oriented to generate one channel
    images. So, despite you can change the brush color,
    the output data is a binary black and white image
    (pixel values for color = 255.0 and not color = 0.0).

    When you save an image, a thumbnail appears below
    the canvas so you can see a collection of all the
    data you have already created.

    Google Colab is not yet suported (but I know how
    to do it :).

    Want to learn something about Python? If you do
    not extend this class from `object` and you
    use the `autoreload` module, this class is not
    going to work because the autoreload module
    reloads the class each time it is used and this
    clears the `_instances` static attribute.

    Attributes:
        canvas_id (str): ID of the generated HTML canvas.
        rendered (bool): Flag to check if the canvas has
            been already been showed. You cannot construct
            two canvas with the same ID in the same page.
            If you do it you will lose the ability to paint
            in all others canvas with the same ID created
            before. Of course you can solve this creating
            an unique id of each of the canvas instances of
            the same object but, it is not very useful and
            I am too lazy to implement that.
    """

    _instances = {}

    def __init__(self, canvas_w, canvas_h, out_w, out_h,
                 brush_width=None, brush_color="black",
                 data=None, title=""):
        """Create a new NotebookPainter.

        Args:
            canvas_w (int): Width of the HTML canvas in pixels.
            canvas_h (int): Height of the HTML canvas in pixels.
            out_w (int): Width of the resulting array.
                See :func:`NotebookPainter.get_data` and
                :func:`NotebookPainter.get_numpy_data`.
            out_h (int): Height of the resulting array.
                See :func:`NotebookPainter.get_data` and
                :func:`NotebookPainter.get_numpy_data`.
            brush_width (int): Width of the brush used to paint.
                If brush_width` is `None`, brush width is set to
                15% of the minimun between canvas width and height.
                Default: `None`.
            brush_color (str): Color of the brush.
                Default: `"black"`.
            data (list of lists): Create a NotebookPainter with
                existing data. At this moment, the existing images
                are not show below the canvas but you will see the
                new images you add after the creation.
                Default: `None`.
            title (str): Text that will be show on the header
                of the canvas.
                Default: `""`.
        """
        if brush_width is None:
            brush_width = int(min(canvas_w, canvas_h) * 0.15)
        self.canvas_id = str(uuid.uuid4())
        self.rendered = False
        self._data = [] if data is None else data
        # Save instance in a class-level dictionary.
        NotebookPainter._instances[self.canvas_id] = self
        # Concatenate code of the widget.
        template = _js_code + _css_code + _html_code
        # Define a dict with params to apply in the template.
        params = {
            "id": self.canvas_id,
            "title": title,
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "out_w": out_w,
            "out_h": out_h,
            "brush_width": brush_width,
            "brush_color": brush_color
        }
        # Apply template.
        self._html = apply_template(template, **params)

    def get_data(self):
        """Return a list with pixel data. """
        return self._data

    def get_numpy_data(self):
        """Return a NumPy array with pixel data.

        Returns:
            `numpy.ndarray` of floats with the pixel data
            of the imagenes you have created until this
            moment.
        """
        return np.array(self._data).astype("float")

    def _repr_html_(self):
        """Render your canvas if it has not already been rendered. """
        if not self.rendered:
            self.rendered = True
            return self._html
        raise Exception("This canvas had already been rendered. "
                        "Reusing canvas is not implemented yet, "
                        "create a new one. "
                        "If you want to reuse painted data use the data param "
                        "in the constructor.")

