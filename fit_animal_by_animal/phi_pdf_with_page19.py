# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfReader
from time_vary_norm_utils import phi_t_fn

BATCHES = ["Comparable", "SD"]
RESULTS_DIR = os.path.dirname(__file__)
MODEL_KEY = "vbmc_time_vary_norm_tied_results"
PARAM_MAP = {
    "h1": "bump_width_samples",
    "a1": "bump_height_samples",
    "h2": "dip_width_samples",
    "a2": "dip_height_samples",
    "b1": 0.0
}
T_RANGE = np.linspace(0, 1, 200)

for batch in BATCHES:
    pkl_files = [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith(f"results_{batch}_animal_") and f.endswith(".pkl")
    ]
    animal_ids = [
        int(f.split("_")[-1].replace(".pkl", "")) for f in pkl_files
    ]
    animal_ids, pkl_files = zip(*sorted(zip(animal_ids, pkl_files))) if animal_ids else ([],[])
    n_animals = len(animal_ids)
    if n_animals == 0:
        print(f"No animals found for batch {batch}")
        continue
    out_pdf_path = os.path.join(RESULTS_DIR, f"phi_and_page19_{batch}.pdf")
    with PdfPages(out_pdf_path) as pdf_out:
        for animal_id, pkl_file in zip(animal_ids, pkl_files):
            # 1. Read page 19 from corresponding pdf
            pdf_file = pkl_file.replace('.pkl', '.pdf')
            pdf_path = os.path.join(RESULTS_DIR, pdf_file)
            fig, axs = plt.subplots(2, 1, figsize=(8.5, 11))
            # --- Top: Page 19 from animal's PDF (rendered as image) ---
            axs[0].axis('off')
            if os.path.exists(pdf_path):
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(pdf_path)
                    if len(doc) > 18:
                        page = doc[18]
                        pix = page.get_pixmap(dpi=150)
                        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        if img.shape[2] == 4:
                            img = img[..., :3]  # Drop alpha if present
                        axs[0].imshow(img)
                        axs[0].set_title(f"Page 19 of PDF: Animal {animal_id}")
                        axs[0].axis('off')
                    else:
                        axs[0].text(0.5, 0.5, f"PDF has only {len(doc)} pages", ha='center')
                except ImportError:
                    axs[0].text(0.5, 0.5, "PyMuPDF (fitz) not installed. Please install with 'pip install pymupdf' to enable PDF page rendering.", ha='center')
                except Exception as e:
                    axs[0].text(0.5, 0.5, f"PDF error: {e}", ha='center')
            else:
                axs[0].text(0.5, 0.5, "PDF not found", ha='center')

            # --- Bottom: phi(t) plot ---
            pkl_path = os.path.join(RESULTS_DIR, pkl_file)
            try:
                with open(pkl_path, 'rb') as f:
                    results = pickle.load(f)
                if MODEL_KEY in results:
                    model_res = results[MODEL_KEY]
                    phi_args = {k: np.mean(model_res[v]) if v in model_res else 0.0 for k, v in PARAM_MAP.items() if k != 'b1'}
                    phi_args['b1'] = PARAM_MAP['b1']
                    phi_vals = phi_t_fn(T_RANGE, **phi_args)
                    axs[1].plot(T_RANGE, phi_vals)
                    axs[1].set_xlabel('t')
                    axs[1].set_ylabel('phi(t)')
                    axs[1].set_title(f"phi(t) for Animal {animal_id}")
                else:
                    axs[1].text(0.5, 0.5, "Model not found in pickle", ha='center')
                    axs[1].axis('off')
            except Exception as e:
                axs[1].text(0.5, 0.5, f"Pickle error: {e}", ha='center')
                axs[1].axis('off')
            plt.tight_layout()
            pdf_out.savefig(fig)
            plt.close(fig)
    print(f"Saved: {out_pdf_path}")
