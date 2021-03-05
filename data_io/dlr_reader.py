from re import compile
pattern_read = compile(r'(^\w*).(\w*):\s*([\w\d,\[\]\.]*)')
pattern_write = compile(r'.*\[([\d\.]*),([\d\.]*),([\d\.]*)\]')


def ReadAux_DLR(aux_path):
    image_dat = {}
    with open(aux_path, "r") as meta:
        for line in meta.readlines():
            try:
                item, prop, value = pattern_read.findall(line)[0]
                if item not in image_dat:
                    image_dat[item] = {}
                image_dat[item][prop] = value
            except ValueError:
                pass

    return image_dat