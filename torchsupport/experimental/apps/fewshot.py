from flexx import flx
from flexx import event
import os

from tornado.web import StaticFileHandler

class ScaleImageWidget(flx.Widget):
    """ Display an image from a url.
    
    The ``node`` of this widget is an
    `<img> <https://developer.mozilla.org/docs/Web/HTML/Element/img>`_
    wrapped in a `<div> <https://developer.mozilla.org/docs/Web/HTML/Element/div>`_
    (the ``outernode``) to handle sizing.
    """

    DEFAULT_MIN_SIZE = 16, 16

    _sequence = 0

    source = event.StringProp('', settable=True, doc="""
        The source of the image, This can be anything that an HTML
        img element supports.
        """)

    stretch = event.BoolProp(False, settable=True, doc="""
        Whether the image should stretch to fill all available
        space, or maintain its aspect ratio (default).
        """)

    def _create_dom(self):
        global window
        outer = window.document.createElement('div')
        inner = window.document.createElement('img')
        outer.appendChild(inner)
        return outer, inner

    @event.reaction
    def __resize_image(self):
        size = self.size
        if self.stretch:
          self.node.style.maxWidth = None
          self.node.style.maxHeight = None
          self.node.style.width = size[0] + 'px'
          self.node.style.height = size[1] + 'px'
        else:
          self.node.style.backgroundColor = None
          self.node.style.marginLeft = "5%"
          self.node.style.marginTop = "5%"
          self.node.style.maxWidth = "90%"
          self.node.style.maxWidth = "auto"
          self.node.style.width = "90%"
          self.node.style.height = "auto"

    @event.reaction
    def __source_changed(self):
      self.node.src = self.source

class ClickableImage(flx.Widget):
  def init(self, source):
    self.src = source
    self.img = ScaleImageWidget(source = source, flex=1)
    self.img.node.addEventListener("mouseover",
      lambda e: self._show_clickable_in())
    self.img.node.addEventListener("mouseout",
      lambda e: self._show_clickable_out())

  def _show_clickable_in(self):
    size = self.img.size[0]
    p20 = size // 20
    self.img.node.style.boxShadow = "0px 0px "+ p20 + "px 2px black"

  def _show_clickable_out(self):
    self.img.node.style.boxShadow = None

  @flx.action
  def set_source(self, source):
    self.src = source
    if self.src == None:
      self.img.node.style.visibility = "hidden"
    else:
      self.img.node.style.visibility = "visible"
      self.img.set_source(source)

class ImageGrid(flx.Widget):
  def init(self, width=4, height=4,
           path=lambda x, y: "/images/starting_image.png",
           handler=lambda o, x, y: print(x, y)):
    self.width = width
    self.height = height
    self.path = path
    self.handler = handler
    self.imageGrid = [[None for idy in range(height)] for idx in range(width)]
    with flx.HFix():
      for idx in range(width):
        with flx.VFix(flex=1):
          for idy in range(height):
            self.imageGrid[idx][idy] = ClickableImage(path(idx, idy), flex=1)
            a, b = idx, idy
            self.imageGrid[idx][idy].node.addEventListener("click",
              self._on_click_handler(a, b))

  def _on_click_handler(self, idx, idy):
    return lambda e: self.handler(self, idx, idy)

def path_provider(x, y):
  if (x + y) % 2 == 0:
    return "/images/starting_image.png"
  else:
    return "/images/cytosol_image.png"

class FewShot(flx.Widget):
  def init(self):
    self.selectedImages = []

    with flx.TabLayout() as self.tabs:
      with flx.HFix(title="selection", flex=1) as self.selector_view:
        with flx.VFix() as self.images:
          flx.Label(text="Images", flex=(1, 1))
          self.imageGrid = ImageGrid(4, 4, path_provider,
                                    lambda o, idx, idy: self.image_click_handler(o, idx, idy),
                                    flex=(1, 9))
        self.images.node.style.backgroundColor = "#88888888"
        with flx.VFix() as self.selected:
          flx.Label(text="Selected", flex=(1, 1))
          self.selectedGrid = ImageGrid(4, 4, self.selected_provider,
                                        lambda o, idx, idy: self.selected_click_handler(o, idx, idy),
                                        flex=(1, 9))
      with flx.HFix(title="results", flex=1) as self.result_view:
        self.resultGrid = ImageGrid(8, 4, path_provider,
                                    flex=(1, 1))

  @flx.action
  def image_click_handler(self, o, idx, idy):
    source = o.imageGrid[idx][idy].src
    if (source, idx, idy) not in self.selectedImages:
      self.selectedImages.append((source, idx, idy))
      length = len(self.selectedImages)
      new_position = (
        (length - 1) % 4,
        (length - 1) // 4
      )
      self.selectedGrid.imageGrid[new_position[0]][new_position[1]].set_source(source)

  @flx.action
  def selected_click_handler(self, o, idx, idy):
    position = idy * 4 + idx
    if position < len(self.selectedImages):
      self.selectedImages.pop(position)
      self.selectedGrid.imageGrid[idx][idy].set_source(None)
      for pos, elem in enumerate(self.selectedImages):
        source = elem[0]
        new_position = (
          pos % 4,
          pos // 4
        )
        self.selectedGrid.imageGrid[new_position[0]][new_position[1]].set_source(source)
      for pos in range(len(self.selectedImages), 16):
        new_position = (
          pos % 4,
          pos // 4
        )
        self.selectedGrid.imageGrid[new_position[0]][new_position[1]].set_source(None)

  def selected_provider(self, idx, idy):
    return lambda x, y: None

tornado_app = flx.create_server().app
dirname = os.path.expanduser('~/Documents/knoplab/yeastimages_presentation/')
tornado_app.add_handlers(r".*", [
    (r"/images/(.*)", StaticFileHandler, {"path": dirname}),
])
app = flx.App(FewShot)
app.launch('browser')
flx.run()
