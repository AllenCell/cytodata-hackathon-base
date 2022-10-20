import numpy as np
import copy

num_to_protein_full = {}
num_to_protein_full[0] = "nucleoplasm"
num_to_protein_full[1] = "nuclear membrane"
num_to_protein_full[2] = "nucleoli"
num_to_protein_full[3] = "nucleoli fibrillar center"
num_to_protein_full[4] = "nuclear speckles"
num_to_protein_full[5] = "nuclear bodies"
num_to_protein_full[6] = "endoplasmic reticulum"
num_to_protein_full[7] = "golgi apparatus"
num_to_protein_full[8] = "peroxisomes"
num_to_protein_full[9] = "endosomes"
num_to_protein_full[10] = "lysosomes"
num_to_protein_full[11] = "intermediate filaments"
num_to_protein_full[12] = "actin filaments"
num_to_protein_full[13] = "focal adhesion sites"
num_to_protein_full[14] = "microtubules"
num_to_protein_full[15] = "microtubule ends"
num_to_protein_full[16] = "cytokinetic bridge"
num_to_protein_full[17] = "mitotic spindle"
num_to_protein_full[18] = "microtubule organizing center"
num_to_protein_full[19] = "centrosome"
num_to_protein_full[20] = "lipid droplets"
num_to_protein_full[21] = "plasma membrane"
num_to_protein_full[22] = "cell junctions"
num_to_protein_full[23] = "mitochondria"
num_to_protein_full[24] = "aggresome"
num_to_protein_full[25] = "cytosol"
num_to_protein_full[26] = "cytoplasmic bodies"
num_to_protein_full[27] = "rods & rings"

num_to_protein_single_cells = {}
num_to_protein_single_cells[0] = "nucleoplasm"
num_to_protein_single_cells[1] = "nuclear membrane"
num_to_protein_single_cells[2] = "nucleoli"
num_to_protein_single_cells[3] = "nucleoli fibrillar center"
num_to_protein_single_cells[4] = "nuclear speckles"
num_to_protein_single_cells[5] = "nuclear bodies"
num_to_protein_single_cells[6] = "endoplasmic reticulum"
num_to_protein_single_cells[7] = "golgi apparatus"
num_to_protein_single_cells[8] = "intermediate filaments"
num_to_protein_single_cells[9] = "actin filaments,focal adhesion sites"
num_to_protein_single_cells[10] = "microtubules"
num_to_protein_single_cells[11] = "mitotic spindle"
num_to_protein_single_cells[12] = "centrosome,centriolar satellite"
num_to_protein_single_cells[13] = "plasma membrane,cell junctions"
num_to_protein_single_cells[14] = "mitochondria"
num_to_protein_single_cells[15] = "aggresome"
num_to_protein_single_cells[16] = "cytosol"
num_to_protein_single_cells[
    17
] = "vesicles,peroxisomes,endosomes,lysosomes,lipid droplets,cytoplasmic bodies"
num_to_protein_single_cells[18] = "no staining"

whole2single = {
    "nucleoplasm": 0,
    "nuclear membrane": 1,
    "nucleoli": 2,
    "nucleoli fibrillar center": 3,
    "nuclear speckles": 4,
    "nuclear bodies": 5,
    "endoplasmic reticulum": 6,
    "golgi apparatus": 7,
    "intermediate filaments": 8,
    "actin filaments": 9,
    "focal adhesion sites": 9,
    "microtubules": 10,
    "microtubule ends": 10,
    # "microtubule organizing center": 10,
    # "cytokinetic bridge": 10,
    "mitotic spindle": 11,
    "mitotic chromosome": 11,
    "centrosome": 12,
    "centriolar satellite": 12,
    "plasma membrane": 13,
    "cell junctions": 13,
    "mitochondria": 14,
    "aggresome": 15,
    "cytosol": 16,
    "vesicles": 17,
    "peroxisomes": 17,
    "endosomes": 17,
    "lysosomes": 17,
    "lipid droplets": 17,
    "cytoplasmic bodies": 17,
    "no staining": 18,
    "rods & rings": 18,
    "nucleoli rim": 18,
    "kinetochore": 18,
}

protein_to_num_full = {v.lower(): k for (k, v) in num_to_protein_full.items()}
protein_to_num_single_cells = {
    v.lower(): k for (k, v) in num_to_protein_single_cells.items()
}

num_whole2single = {
    protein_to_num_full[k]: v
    for k, v in whole2single.items()
    if k in protein_to_num_full
}

num_single2whole = {
    v: protein_to_num_full[k]
    for k, v in whole2single.items()
    if k in protein_to_num_full
}

protein_to_num_ref = copy.deepcopy(protein_to_num_full)
protein_to_num_ref["vesicles"] = 28
protein_to_num_ref["unknown"] = 29
protein_to_num_ref["nucleoli rim"] = 30
protein_to_num_ref["mitotic chromosome"] = 31
protein_to_num_ref["kinetochore"] = 32

num_to_protein_4k = {}
num_to_protein_4k[0] = "nucleoplasm"
num_to_protein_4k[1] = "plasma membrane"
num_to_protein_4k[2] = "mitochondria"
num_to_protein_4k[3] = "cytosol"

protein_to_num_4k = {v.lower(): k for (k, v) in num_to_protein_4k.items()}

num_to_protein_5k = {}
num_to_protein_5k[0] = "nucleoplasm"
num_to_protein_5k[1] = "plasma membrane"
num_to_protein_5k[2] = "mitochondria"
num_to_protein_5k[3] = "cytosol"
num_to_protein_5k[4] = "vesicles"

protein_to_num_5k = {v.lower(): k for (k, v) in num_to_protein_5k.items()}

num_to_other_protein_5k = {}
num_to_other_protein_5k[0] = "golgi apparatus"
num_to_other_protein_5k[1] = "nuclear speckles"
num_to_other_protein_5k[2] = "nuclear bodies"
num_to_other_protein_5k[3] = "nucleoli"
num_to_other_protein_5k[4] = "endoplasmic reticulum"

other_protein_to_num_5k = {v.lower(): k for (k, v) in num_to_other_protein_5k.items()}
