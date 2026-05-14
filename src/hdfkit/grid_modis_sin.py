from dataclasses import field, dataclass
from pathlib import Path

def get_grid_hv_surround(hv:str, direction:str):
    h = int(hv[1:3])
    v = int(hv[4:6])
    match (direction):
        case "left":
            _h, _v = h - 1, v
        case "right":
            _h, _v = h + 1, v
        case "top":
            _h, _v = h, v - 1
        case "bottom":
            _h, _v = h, v + 1
        case "topleft":
            _h, _v = h - 1, v - 1
        case "topright":
            _h, _v = h + 1, v - 1
        case "bottomleft":
            _h, _v = h - 1, v + 1
        case "bottomright":
            _h, _v = h + 1, v + 1
        case _:
            raise ValueError(f"Invalid direction: {direction}")
    match (_h):
        case -1:
            _h = 35
        case 36:
            _h = 0
        case _:
            pass
    match (_v):
        case -1:
            _v = 17
        case 18:
            _v = 0
        case _:
            pass
    return f"h{_h:02d}v{_v:02d}"


   

@dataclass
class TileGridModisSin:
    gcenter: str
    gsize: int = 1200
    gleft: str = field(init=False)
    gright: str = field(init=False)
    gtop: str = field(init=False)
    gbottom: str = field(init=False)
    gtopleft: str = field(init=False)
    gtopright: str = field(init=False)
    gbottomleft: str = field(init=False)
    gbottomright: str = field(init=False)
    fcenter: str = field(init=True, default=None)
    fleft: str = field(init=False, default=None)
    fright: str = field(init=False, default=None)
    ftop: str = field(init=False, default=None)
    fbottom: str = field(init=False, default=None)
    ftopleft: str = field(init=False, default=None)
    ftopright: str = field(init=False, default=None)
    fbottomleft: str = field(init=False, default=None)
    fbottomright: str = field(init=False, default=None)
    do_grid_surround: bool = field(init=True, default=True)

    def __post_init__(self):
        if not isinstance(self.gcenter, str):
            raise ValueError("gcenter must be a string")
        if self.do_grid_surround:
            self.gleft = get_grid_hv_surround(self.gcenter, "left")
            self.gright = get_grid_hv_surround(self.gcenter, "right")
            self.gtop = get_grid_hv_surround(self.gcenter, "top")
            self.gbottom = get_grid_hv_surround(self.gcenter, "bottom")
            self.gtopleft = get_grid_hv_surround(self.gcenter, "topleft")
            self.gtopright = get_grid_hv_surround(self.gcenter, "topright")
            self.gbottomleft = get_grid_hv_surround(self.gcenter, "bottomleft")
            self.gbottomright = get_grid_hv_surround(self.gcenter, "bottomright")
            if self.fcenter is not None:
                if not Path(self.fcenter).exists():
                    raise FileNotFoundError(f"File {self.fcenter} not found")
                if self.gcenter.lower() in (Path(self.fcenter).name).lower():
                    self.replace_hv_surround()
    
    def replace_hv_surround(self):
        fcenter = Path(self.fcenter).name
        islower = self.gcenter.lower() in fcenter
        gc, gl, gr, gt, gb, gtl, gtr, gbl, gbr = self.gcenter, self.gleft, self.gright, self.gtop, self.gbottom, self.gtopleft, self.gtopright, self.gbottomleft, self.gbottomright
        fc, fl, fr, ft, fb, ftl, ftr, fbl, fbr = self.fcenter, self.fleft, self.fright, self.ftop, self.fbottom, self.ftopleft, self.ftopright, self.fbottomleft, self.fbottomright
        if islower:
            fr = Path(fc).with_name(fcenter.replace(gc.lower(), gr.lower()))
            fl = Path(fc).with_name(fcenter.replace(gc.lower(), gl.lower()))
            ft = Path(fc).with_name(fcenter.replace(gc.lower(), gt.lower()))
            fb = Path(fc).with_name(fcenter.replace(gc.lower(), gb.lower()))
            ftl = Path(fc).with_name(fcenter.replace(gc.lower(), gtl.lower()))
            ftr = Path(fc).with_name(fcenter.replace(gc.lower(), gtr.lower()))
            fbl = Path(fc).with_name(fcenter.replace(gc.lower(), gbl.lower()))
            fbr = Path(fc).with_name(fcenter.replace(gc.lower(), gbr.lower()))
        else:
            fr = Path(fc).with_name(fcenter.replace(gc.upper(), gr.upper()))
            fl = Path(fc).with_name(fcenter.replace(gc.upper(), gl.upper()))
            ft = Path(fc).with_name(fcenter.replace(gc.upper(), gt.upper()))
            fb = Path(fc).with_name(fcenter.replace(gc.upper(), gb.upper()))
            ftl = Path(fc).with_name(fcenter.replace(gc.upper(), gtl.upper()))
            ftr = Path(fc).with_name(fcenter.replace(gc.upper(), gtr.upper()))
            fbl = Path(fc).with_name(fcenter.replace(gc.upper(), gbl.upper()))
            fbr = Path(fc).with_name(fcenter.replace(gc.upper(), gbr.upper()))
        self.fleft = fl.as_posix() if fl.exists() else None
        self.fright = fr.as_posix() if fr.exists() else None
        self.ftop = ft.as_posix() if ft.exists() else None
        self.fbottom = fb.as_posix() if fb.exists() else None
        self.ftopleft = ftl.as_posix() if ftl.exists() else None
        self.ftopright = ftr.as_posix() if ftr.exists() else None
        self.fbottomleft = fbl.as_posix() if fbl.exists() else None
        self.fbottomright = fbr.as_posix() if fbr.exists() else None
