import numpy as np
from numpy.polynomial.chebyshev import chebval
import matplotlib.pyplot as plt
from scm.plams import *  # source: https://www.scm.com/doc/PythonExamples/xrd/index.html
from ase import Atoms
from pymatgen.core.structure import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class Model:
    def __init__(self):
        self.two_theta = None
        self.intensities = None

    def plot(self, label=""):
        # Normalizamos la intensidad a 100 para que encaje con la teórica
        intensity_norm = (self.intensities / np.max(self.intensities)) * 100
        plt.plot(self.two_theta, intensity_norm, label=label)
        plt.xlabel("2θ")
        plt.ylabel("Intensidad Relativa (%)")


class ExperimentalModel(
    Model
):  # Contiene los datos del modelo experimental y calcula lo que depende del mismo

    def __init__(self, xrdml: str):  # se activa recibiendo un archivo .xrdml

        super().__init__()
        self.xrdml = xrdml
        self.intensities = self.int_extraction()
        self.two_theta = self.theta_extraction()

    def int_extraction(self):
        intensities = []
        with open(self.xrdml, "rt") as xrdml:
            for line in xrdml:
                if '<intensities unit="counts">' in line:
                    # <intensities unit="counts"> 0 1 0 2 ... </dataPoints>
                    line = line.split(">", 1)[1]
                    line = line.split("</")[0]
                    values = line.split()
                    values = [float(value) for value in values]
                    intensities.extend(values)
        return np.array(intensities)

    def theta_extraction(self):
        theta_min = None
        theta_max = None

        with open(self.xrdml, "r") as xrdml:
            for line in xrdml:
                # Si encontramos la etiqueta de inicio, sacamos el número
                if "<startPosition>" in line:
                    theta_min = float(line.split("<startPosition>")[1].split("</")[0])

                # Si encontramos la etiqueta de fin, sacamos el número
                elif "<endPosition>" in line:
                    theta_max = float(line.split("<endPosition>")[1].split("</")[0])

                # Si ya tenemos los dos, podemos romper el bucle para no leer el resto del archivo
                if theta_min is not None and theta_max is not None:
                    break

        # Generamos el array
        two_theta = np.linspace(theta_min, theta_max, len(self.intensities))
        return np.array(two_theta)

    def get_weights(
        self,
    ) -> (
        np.ndarray
    ):  # obtenemos los pesos como la inversa de la varianza da intensidad experimental y los metemos en un array
        i = self.intensities
        weights = np.where(
            i > 0, 1.0 / i, 0.0
        )  # Si se cumple la condición, elemento = x. Si no, elemento = y
        return np.array(weights)


class TheoreticalModel(
    Model
):  # Contiene los datos del modelo teórico y calcula lo que depende de estos

    def __init__(self, cif: str):  # se activa recibiendo un archivo .cif
        self.cif = cif
        self.active_params = ["scale"]
        self.set_starting_parameters()
        self.wavelength = self.xrd_calc = XRDCalculator(
            wavelength="CuKa"
        )
        
        self.bounds = {
            "B_overall": (0.1, 3.0),
            "U": (0.0, 0.02),
            "W": (1e-6, None),
            "X": (0.0, 0.05),
            "Y": (0.001, 0.5),
            "MD_r": (0.1, 5.0)
        }

        super().__init__()
        self.two_theta, self.intensities = self.get_bragg_peaks()

    def get_bragg_peaks(self):

        self.structure = Structure.from_file(self.cif)
        pattern = self.xrd_calc.get_pattern(self.structure)
        self.hkl = [[equiv['hkl'] for equiv in reflexion] for reflexion in pattern.hkls]

        return pattern.x, pattern.y  # 2theta, intensidad

    def plot(self, label="Teórico"):
        # Usamos vlines (líneas verticales) para los picos
        plt.vlines(self.two_theta, 0, self.intensities, colors="r", linestyles="solid")

    def set_starting_parameters(self) -> None:

        info = []
        reading_info = False
        with open(self.cif, "rt") as cif:
            for line in cif:

                # 1) Parámetros de celda
                if "_cell_length_a" in line:
                    self.a = float(line.split("a")[1].split("(")[0])
                if "_cell_length_b" in line:
                    self.b = float(line.split("b")[1].split("(")[0])
                if "_cell_length_c" in line:
                    self.c = float(line.split("th_c")[1].split("(")[0])
                if "_cell_angle_alpha" in line:
                    self.alpha = float(line.split("alpha")[1].split("(")[0])
                if "_cell_angle_beta" in line:
                    self.beta = float(line.split("beta")[1].split("(")[0])
                if "_cell_angle_gamma" in line:
                    self.gamma = float(line.split("gamma")[1].split("(")[0])

                # 2) Factor de ocupación, coordenadas atómicas fraccionarias
                #  y factores de desplazamiento atómico

                if reading_info:
                    if "loop_" in line:
                        break
                    info.append(line)

                if (
                    "_atom_site_type_symbol" in line
                ):  # Si alguna vez refinas algo que no sea scheelite, revisa el cif y que la línea sea esa
                    reading_info = True

            # for i in range(1, min(len(info), 4)):
            # atom_i = info[i].split()
            # sof = float(atom_i[1].split("(")[0])  # SOF = Factor de ocupación
            # x = float(atom_i[2].split("(")[0])
            # y = float(atom_i[3].split("(")[0])
            # z = float(atom_i[4].split("(")[0])
            # u_iso = float(atom_i[6].split("(")[0])  # U_iso or equivalent
            # setattr(self, f"sof_{i}", sof)
            # setattr(self, f"x_{i}", x)
            # setattr(self, f"y_{i}", y)
            # setattr(self, f"z_{i}", z)
            # setattr(self, f"u_iso_{i}", u_iso)

        # Estos son todos los parámetros refinables que podemos extraer del CIF
        # Vamos ahora a obtener los que no se pueden extraer del CIF

        # 3) Factor de escala. Se define como el cociente entre las intensidades máximas
        # experimental y teórica

        self.scale = 1.0
        # realmente se define como el cociente entre los máximos de las intensidades experimental (numerador) y teórica (denominador),
        # pero por ahora pondremos esto y actualizaremos el dato en RietveldSolver para no llamar métodos de ExperimentalModel en TheoreticalModel

        # 4) Error de cero angular y de la altura de la muestra
        self.two_theta_zero = 0.0
        self.sample_displacement = 0.0

        # 5) Perfil de pico
        self.U = 0.001
        self.V = 0.0
        self.W = 0.01
        self.X = 0.001
        self.Y = 0.001
        self.MD_r = 1.0  # (Orientación preferente, March-Dollase)

        # 6) Coeficientes de Chebyshev. Como mucho se pueden tomar 6 para evitar overfitting.
        self.b0 = 100  # se suele tomar min(I_exp). También actualizaremos este parámetro en RietveldSolver
        self.b1 = self.b2 = self.b3 = self.b4 = self.b5 = (
            0  # Se empieza en 0 y el algoritmo los va cambiando para curvar la línea que modela el fondo
        )

        # 7) Fondo de aire.
        self.B_air = 100.0

        # 8) Constante de Debye-Waller.
        self.B_overall = 0.75

    def calc_march_dollase(self, cos_alpha: float, r: float) -> float:
        # Blindaje contra valores r negativos o cero
        r_safe = np.maximum(r, 1e-4) 
        cos_sq = cos_alpha**2
        sin_sq = 1 - cos_sq
        
        term = (r_safe**2 * cos_sq) + ((1.0 / r_safe) * sin_sq)
        return term**(-1.5)

    def get_free_cell_parameters(
        self,
    ) -> (
        list
    ):  # Determina qué parámetros de celda son independientes según la simetría.

        sga = SpacegroupAnalyzer(self.structure)
        self.system = (
            sga.get_crystal_system()
        )  # Devuelve la estructura ("cubic", "monoclinic", ...)

        mapping = {
            "cubic": ["a"],
            "tetragonal": ["a", "c"],
            "hexagonal": ["a", "c"],
            "trigonal": ["a", "c"],
            "orthorhombic": ["a", "b", "c"],
            "monoclinic": ["a", "b", "c", "beta"],
            "triclinic": ["a", "b", "c", "alpha", "beta", "gamma"],
        }

        return mapping.get(
            self.system, ["a", "c"]  # Por defecto, tetragonal (Scheelite)
        )  # Basta añadir el string sumándolo a p en el refine() para incluir estos parámetros en el refinamiento

    def set_active_parameters(
        self, active_params: list
    ):  # Decide qué parámetros se refinan en cada stage
        self.active_params = active_params  # Llamaremos a este método desde refine, donde se establecerá qué parámetros deben estar activos

    def get_parameters(self):  # Empaqueta los parámetros activos
        p = []
        for name in self.active_params:
            p.append(
                getattr(self, name)
            )  # Empareja un string con una variable que se llame igual. Así empaquetamos los parámetros
        return np.array(p)

    def update_parameters(self, p: np.ndarray) -> np.ndarray:
        # desempaqueta los números sin física que devuelve cada iteración y les dota de sentido

        magnitudes = []
        for name, value in zip(self.active_params, p):
            # Si el parámetro tiene límites definidos, recortamos el valor numérico
            if name in self.bounds:
                low, high = self.bounds[name]
                if low is not None:
                    value = max(low, value)
                if high is not None:
                    value = min(high, value)
            setattr(self, name, value)
            magnitudes.append(
                name
            )  # vector con todos los parámetros actualizados. Guarda los nombres físicos de las variables.

        system = self.system.lower()  # Pone todo en minúsculas por si las moscas
        if system == "tetragonal":
            self.b = (
                self.a
            )  # Para que el programa te devuelva el b refinado y no el del cif.
        elif system == "cubic":
            self.b = self.a
            self.c = self.a
        elif system in ["trigonal", "rhombohedral", "hexagonal"]:
            # Detectamos si usa ejes hexagonales (gamma = 120)
            if abs(self.gamma - 120.0) < 1.0:
                # Aplica a Hexagonal puro o Romboédrico/Trigonal en ajuste hexagonal
                self.b = self.a
                self.alpha = 90.0
                self.beta = 90.0
                self.gamma = 120.0
            else:
                # Aplica a Romboédrico primitivo (a=b=c, alpha=beta=gamma)
                self.b = self.a
                self.c = self.a
                self.beta = self.alpha
                self.gamma = self.alpha

        # Comprobamos si se está refinando algún parámetro de red. En caso afirmativo, actualizamos la estructura para que el programa no coja los parámetros
        # de red iniciales.
        lattice_params = ["a", "b", "c", "alpha", "beta", "gamma"]
        if any(param in self.active_params for param in lattice_params):

            # 1) Crear red
            new_lattice = Lattice.from_parameters(
                self.a, self.b, self.c, self.alpha, self.beta, self.gamma
            )
            # 2) Reemplazar estructura
            self.structure = Structure(
                new_lattice, self.structure.species, self.structure.frac_coords
            )
            # 3) Recalculamos los picos de Bragg con la nueva celda
            pattern = self.xrd_calc.get_pattern(self.structure)
            self.two_theta = pattern.x
            self.intensities = pattern.y

        return np.array(magnitudes)

    def calc_cell_volume(self) -> float:
        # Calcula el volumen del caso más general: celda triclínica. Los casos particulares se cubren solo
        # 1. Convertir ángulos de grados a radianes
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)

        # 2. Calcular los cosenos
        ca = np.cos(alpha_rad)
        cb = np.cos(beta_rad)
        cg = np.cos(gamma_rad)

        # 3. Aplicar la fórmula triclínica
        term_under_sqrt = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg

        cell_volume = self.a * self.b * self.c * np.sqrt(term_under_sqrt)
        self.cell_volume = cell_volume  # Lo guardamos como atributo por si hace falta

        return self.cell_volume

    def calc_crystal_size(self) -> float:
        # Se obtiene a partir de la ecuación de Scherrer
        K = 0.9  # Asumimos partículas esferoidales, que son las que nos interesan.
        lambda_cu = 1.5406

        if abs(self.Y) < 1e-6:
            return float("inf")  # Tamaño infinito si no hay ensanchamiento
        Y = (
            self.Y
        )  # Cuando invoquemos calc_crystal_size, ya se habrá refinado el modelo, luego en memoria estará el parámetro refinado
        self.crystal_size = K * lambda_cu / (Y * np.pi / 180)  # Scherrer
        return self.crystal_size

    def calc_microstrain(self) -> float:
        # 1. Contribución Gaussiana (U)
        if self.U > 0:
            strain_G = (np.sqrt(self.U) * np.pi / 180.0) / 4.0
        else:
            strain_G = 0.0

        # 2. Contribución Lorentziana (X)
        if self.X > 0:
            strain_L = (self.X * np.pi / 180.0) / 4.0
        else:
            strain_L = 0.0

        # 3. Deformación fraccional absoluta
        self.microstrain = strain_G + strain_L

        return self.microstrain

    def get_physical_report(self) -> dict:
        # asocia cada valor refinado a su correspondiente magnitud física
        self.calc_cell_volume()
        self.calc_crystal_size()
        self.calc_microstrain()
        report_names = {
            "scale": "Factor de escala",
            "a": "Parámetro reticular a (Å)",
            "b": "Parámetro reticular b (Å)",
            "c": "Parámetro reticular c (Å)",
            "alpha": "Ángulo α (deg)",
            "beta": "Ángulo β (deg)",
            "gamma": "Ángulo γ (deg)",
            "MD_r": "Parámetro de MD",
            "cell_volume": "Volumen de celda (Å³)",
            "crystal_size": "Tamaño del cristal (Å)",
            "microstrain": "Microdeformación",
        }
        # Añadimos las propiedades de cada átomo a report_names:
        # i = 1
        # while hasattr(self, f"sof_{i}"):
        # report_names[f"sof_{i}"] = f"SOF del átomo {i}"
        # report_names[f"x_{i}"] = f"coordenada x del átomo {i}"
        # report_names[f"y_{i}"] = f"coordenada y del átomo {i}"
        # report_names[f"z_{i}"] = f"coordenada z del átomo {i}"
        # report_names[f"u_iso_{i}"] = f"Desplazamiento U_iso del átomo {i}"
        # i = i + 1

        report = {}
        for var, name in report_names.items():
            report[name] = getattr(self, var)

        return report


class MinimizationSolver:  # Construcción del algoritmo sin importar la física que subyace

    def __init__(self, residuals_function, jacobian_function):
        self.get_residuals = residuals_function
        self.get_jacobian = jacobian_function
        self.chi2_history = []

    def fit(self, p_initial: np.ndarray, weights: np.ndarray) -> np.ndarray:

        p = p_initial
        max_iter = 50
        tol = 1e-4
        w = weights  # por comodidad
        lam = 0.002  # Source: MathWorks. Suele ser un buen punto de partida, aunque se podría subir a 0.01

        # 1) Recibimos los datos de partida
        res = self.get_residuals(p)
        chi2_current = np.sum(w * res**2)
        self.chi2_history.append(chi2_current)

        for i in range(max_iter):
            # 2) Calculo la jacobiana
            jac = self.get_jacobian(p)

            # 3) Ya tengo los ingredientes para calcular delta_p
            delta_p = self.solve_lm_equation(res, jac, w, lam)

            # 4) Compruebo si el paso es adecuado
            p_test = p + delta_p
            res_test = self.get_residuals(p_test)
            chi2_test = np.sum(w * res_test**2)

            if (
                chi2_test < chi2_current
            ):  # paso satisfactorio. Chi2 debe ser cada vez menor
                delta_chi2 = chi2_current - chi2_test
                if delta_chi2 / chi2_current < tol:
                    # Guardamos el último paso
                    p = p_test
                    break  # convergencia alcanzada
                else:
                    # Actualizamos parámetros
                    p = p_test
                    res = res_test
                    chi2_current = chi2_test
                    self.chi2_history.append(chi2_current)
                    lam = (
                        lam / 10
                    )  # Estamos más cerca del mínimo. Nos interesa que domine G-N
            else:
                lam = (
                    lam * 10
                )  # Nos estaríamos alejando del mínimo. Nos interesa que domine GD

        return p

    def solve_lm_equation(
        self, res: np.ndarray, J: np.ndarray, w: np.ndarray, lam: float
    ) -> np.ndarray:
        w_sqrt = np.sqrt(w)
        wJ = w_sqrt[:, np.newaxis] * J
        wres = w_sqrt * res

        GN_factor = wJ.T @ wJ
        # @ hace la multiplicación matricial, no elemento a elemento
        GD_factor = np.diag(np.diag(GN_factor))
        # Un diag crea un vector 1D. Con dos, formo la matriz diagonal

        eye = np.eye(GN_factor.shape[0])

        res_factor = wJ.T @ wres

        delta_p = np.linalg.solve(GN_factor + lam * GD_factor + 1e-8 * eye, -res_factor)
        # Ecuación de Levenberg-Marquardt

        return delta_p


class RietveldSolver:  # Emplea el algoritmo para resolver el problema de Rietveld específicamente

    def __init__(
        self, experimental_model=ExperimentalModel, theoretical_model=TheoreticalModel
    ):
        self.exp_model = experimental_model
        self.theo_model = theoretical_model

    def calculate_scale(self) -> float:
        i_calc_unscaled = self.calculate_i_calc()
        self.theo_model.scale = np.max(self.exp_model.intensities) / np.max(
            i_calc_unscaled
        )
        return self.theo_model.scale

    def calculate_background(self, two_theta: np.ndarray) -> np.ndarray:
        # Calculamos la línea de background a través de los polinomios de Chebyshev, los cuales son estables en [-1, 1], por lo que debemos normalizar el 2theta primero

        # 1) Normalización de 2theta
        t_min = self.exp_model.two_theta[0]
        t_max = self.exp_model.two_theta[-1]
        x_norm = (2.0 * (self.exp_model.two_theta - t_min) / (t_max - t_min)) - 1.0

        # 2) Extracción de los coeficientes
        coeffs = [
            getattr(self.theo_model, f"b{i}", 0.0) for i in range(6)
        ]  # Genera dinámicamente ['b0', 'b1', ..., 'b5'] y busca en theo_model

        # 3) Cálculo de la base polinómica con chebval
        bg = chebval(x_norm, coeffs)

        # 4) Suma del término de dispersión de aire
        b_air = getattr(self.theo_model, "B_air", 0.0)
        bg += b_air / np.maximum(two_theta, 1e-4)

        return bg

    def pseudo_voigt(self, delta: np.ndarray, H_G: float, H_L: float):
        # delta es la distancia al centro del pico
        # H_G y H_L son las FWHM de la gaussiana y la lorentziana respectivamente

        # Ancho total FWHM estimado (aproximación de Olivero-Longbothum)
        H = (
            H_G**5
            + 2.69269 * H_G**4 * H_L
            + 2.42843 * H_G**3 * H_L**2
            + 4.47163 * H_G**2 * H_L**3
            + 0.07842 * H_G * H_L**4
            + H_L**5
        ) ** 0.2

        if H <= 1e-6:
            H = 1e-6  # Evitamos divergencia. Valores inferiores a 1e-6 favorecen que el programa explote

        r = H_L / H  # Cociente entre ancho Lorentziano y total
        # Polinomio de Thompson-Cox-Hastings para eta
        eta_calc = 1.36603 * r - 0.47719 * (r**2) + 0.11116 * (r**3)

        # Parámetro de mezcla eta. Dicta si domina la componente gaussiana o la lorentziana
        eta = max(0.0, min(1.0, eta_calc))

        # Componente Gaussiana
        g_factor = 4.0 * np.log(2)
        G = (
            (2.0 / H)
            * np.sqrt(np.log(2) / np.pi)
            * np.exp(-g_factor * (delta / H) ** 2)
        )

        # Componente Lorentziana
        L = (2.0 / (np.pi * H)) * (1.0 / (1.0 + 4.0 * (delta / H) ** 2))

        # Función de pseudo - Voigt
        p_voigt = eta * L + (1.0 - eta) * G
        return p_voigt

    def calculate_i_calc(self) -> np.ndarray:
        # Genera el difractograma teórico continuo
        two_theta_exp = self.exp_model.two_theta
        i_calc = self.calculate_background(two_theta_exp)

        # Obtener posiciones e intensidades discretas de Bragg del CIF
        two_theta_bragg = self.theo_model.two_theta
        i_bragg = self.theo_model.intensities
        equiv_hkls = self.theo_model.hkl

        # Parámetros del modelo
        scale = self.theo_model.scale
        B_overall = self.theo_model.B_overall
        wavelength = 1.5406
        zero_shift = self.theo_model.two_theta_zero
        sd = self.theo_model.sample_displacement
        U, V, W = self.theo_model.U, self.theo_model.V, self.theo_model.W
        X, Y = self.theo_model.X, self.theo_model.Y

        # Parámetros adicionales
        a_cell = self.theo_model.a
        c_cell = self.theo_model.c
        MD_r = self.theo_model.MD_r
        H_pref, K_pref, L_pref = 1, 1, 2 # Pico con más cuentas

        # Sumar la contribución de cada pico de Bragg
        for t_k, intensity, equiv_planes in zip(two_theta_bragg, i_bragg, equiv_hkls):

            # Aplicar corrección de cero angular
            theta_rad_k = np.radians(t_k / 2.0)

            # Aplicar atenuación térmica a la intensidad base
            dw_factor = np.exp(
                -2.0 * B_overall * (np.sin(theta_rad_k) / wavelength) ** 2
            )
            corr_intensity = intensity * dw_factor

            pk_sum = 0.0
            # Iteramos sobre todos los planos que caen en este mismo ángulo
            for h, k, l in equiv_planes:

                # Cálculo exacto del ángulo para sistema Tetragonal
                numerador = ((h * H_pref + k * K_pref) / a_cell**2) + ((l * L_pref) / c_cell**2)

                denominador_1 = np.sqrt(((h**2 + k**2) / a_cell**2) + (l**2 / c_cell**2))
                denominador_2 = np.sqrt(((H_pref**2 + K_pref**2) / a_cell**2) + (L_pref**2 / c_cell**2))

                # Evitar divisiones por cero en el pico (0,0,0) si existiera
                if denominador_1 > 0 and denominador_2 > 0:
                    cos_alpha = numerador / (denominador_1 * denominador_2)
                else:
                    cos_alpha = 1.0

                cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
                
                # Factor March - Dollase:
                pk_sum += self.theo_model.calc_march_dollase(cos_alpha, MD_r)
            # Factor March - Dollase promediado en todos los planos equivalentes
            Pk_avg = pk_sum / len(equiv_planes)

            # Corregir aberraciones instrumentales
            t_k_corr = t_k + zero_shift + sd * np.cos(theta_rad_k)

            # Convertir theta a radianes para las funciones trigonométricas de Caglioti
            theta_rad = np.radians(t_k_corr / 2.0)
            tan_th = np.tan(theta_rad)
            cos_th = np.cos(theta_rad)

            # Ancho Gaussiano (Caglioti)
            H_G2 = U * (tan_th**2) + V * tan_th + W
            H_G = np.sqrt(max(1e-6, H_G2))

            # Ancho Lorentziano
            H_L = max(1e-6, X * tan_th + (Y / cos_th if cos_th != 0 else 0))

            # Ventana de corte para no evaluar puntos lejanos innecesariamente (radio = 5*H)
            delta = two_theta_exp - t_k_corr
            cutoff = 5.0 * (H_G + H_L)
            mask = np.abs(delta) <= cutoff

            if np.any(mask):
                profile = self.pseudo_voigt(delta[mask], H_G, H_L)
                i_calc[mask] += (
                    scale * (corr_intensity * Pk_avg) * profile
                )  # Sumamos el fondo a los picos de bragg modelados con pseudo voigt

        return i_calc

    def calculate_residuals(self, p) -> np.ndarray:
        # Los residuos son la diferencia entre la intensidad experimental y calculada punto a punto. Pero no vale la intensidad calculada del cif.
        # Debemos añadir la influencia del fondo, la escala, el error de cero...
        self.theo_model.update_parameters(
            p
        )  # Con esto se obtiene un i_calc nuevo en cada iteración, lo que genera residuos nuevos
        i_exp = self.exp_model.intensities
        i_calc = self.calculate_i_calc()
        residuals = i_exp - i_calc
        return residuals

    def calculate_jacobian(self, p) -> np.ndarray:

        r0 = self.calculate_residuals(
            p
        )  # Array con las funciones que tenemos que derivar
        N = len(
            r0
        )  # Número de filas de la jacobiana = nº de funciones que hay que derivar
        M = len(p)  # Número de columnas de la jacobiana = nº de variables
        J = np.zeros(
            (N, M)
        )  # Definimos primero una matriz vacía y la vamos rellenando mediante iteraciones

        for j in range(M):
            # Reseteamos p en cada iteración para derivar respecto de una variable dejando el resto fijas

            p_perturbed = np.copy(p)

            # Vamos a derivar mediante la definición. Para ello, debemos definir un paso lo suficientemente pequeño:
            h = 1e-5 * max(1, abs(p[j]))
            # Evito /0 y que el paso se acerque al límite de precisión del ordenador
            p_perturbed[j] += h
            res_perturbed = self.calculate_residuals(p_perturbed)

            J[:, j] = (res_perturbed - r0) / h
            # Como no soy matemático esto es un límite con h --> 0

        # Como calculate_residuals() necesita llamar a update_parameters(), está sobreescribiendo theo_model con los parámetros perturbados. Debemos corregirlo
        self.theo_model.update_parameters(p)

        return J

    def calculate_agreement_factors(
        self,
        i_exp: np.ndarray,
        i_calc: np.ndarray,
        w: np.ndarray,
        num_active_params: int,
    ) -> dict:
        # Numerador de minimización y denominador de normalización
        numerator = np.sum(w * (i_exp - i_calc) ** 2)
        denominator = np.sum(w * i_exp**2)

        # 1. R_wp (Weighted Profile)
        r_wp = np.sqrt(numerator / denominator)

        # 2. R_exp (Expected R-factor)
        N = len(i_exp)  # Total de puntos del difractograma
        P = num_active_params  # Grados de libertad consumidos
        r_exp = np.sqrt((N - P) / denominator)

        # 3. Goodness of Fit (Chi-cuadrado reducido)
        chi2 = (r_wp / r_exp) ** 2

        # Devolvemos R_wp y R_exp en formato porcentaje por convención cristalográfica
        return {"R_wp": r_wp * 100, "R_exp": r_exp * 100, "Chi2": chi2}

    def refine(self):  # manda a ejecutar el fit

        # 1) Obtenemos los pesos
        w = self.exp_model.get_weights()

        # 2) Definimos las etapas
        cell_params = self.theo_model.get_free_cell_parameters()
        stages = [
            ["scale", "b0", "b1", "b2", "b3", "b4", "b5", "B_air"],
            ["scale", "b0", "b1", "b2", "b3", "b4", "b5", "B_air", "sample_displacement"],
            ["scale", "b0", "b1", "b2", "b3", "b4", "b5", "B_air", "sample_displacement"]
            + cell_params,
            ["scale", "b0", "b1", "b2", "b3", "b4", "b5", "B_air", "sample_displacement"]
            + cell_params
            + ["B_overall", "MD_r"],
            ["scale", "b0", "b1", "b2", "b3", "b4", "b5", "B_air", "sample_displacement"]
            + cell_params
            + ["B_overall", "MD_r", "X", "Y"],
            ["scale", "b0", "b1", "b2", "b3", "b4", "b5", "B_air", "sample_displacement"]
            + cell_params
            + ["B_overall", "MD_r", "X", "Y", "W"],
            ["scale", "b0", "b1", "b2", "b3", "b4", "b5", "B_air", "sample_displacement"]
            + cell_params
            + ["B_overall", "MD_r", "X", "Y", "W", "U"],
        ]  # Para evitar que el programa se enfade conmigo, me voy a abstener de refinar más cosas por ahora xD

        # 3) Llamamos al motor matemático como objeto
        optimizer = MinimizationSolver(
            residuals_function=self.calculate_residuals,
            jacobian_function=self.calculate_jacobian,
        )

        # 4) Comenzamos a iterar
        for active_params in stages:
            # Activamos los parámetros
            self.theo_model.set_active_parameters(active_params)
            if "scale" in active_params and self.theo_model.scale == 1.0:
                self.theo_model.scale = self.calculate_scale()

            p_initial = self.theo_model.get_parameters()

            # 5) Refinamos los parámetros activos en esta iteración
            p_final = optimizer.fit(p_initial, w)

            # 6) Actualizamos el modelo teórico con los parámeros refinados
            self.theo_model.update_parameters(p_final)
            i_calc_final = self.calculate_i_calc()

            # 7) Comprobamos la evolución de los indicadores
            factors = self.calculate_agreement_factors(
                self.exp_model.intensities, i_calc_final, w, len(active_params)
            )

            # Mostramos el resultado de la etapa por pantalla
            print(f"Fin de la etapa. Parámetros activos: {len(active_params)}")
            print(
                f"Rwp: {factors['R_wp']:.2f}% | Rexp: {factors['R_exp']:.2f}% | Chi2: {factors['Chi2']:.2f}\n"
            )

    def plot_refinement(self):
        x = self.exp_model.two_theta
        y_exp = self.exp_model.intensities
        y_calc = self.calculate_i_calc()
        dif = y_exp - y_calc
        plt.figure(figsize=(6, 10))

        # 1) Curvas experimental y calculada
        plt.plot(x, y_exp, "o", markersize=2, color="black", label="Experimental")
        plt.plot(x, y_calc, "-", color="red", linewidth=1.5, label="Calculada")

        # 2) Curva de diferencia
        offset = (
            -np.max(y_exp) * 0.1
        )  # Baja un poco la curva de diferencia para que no se confunda con el fondo
        plt.plot(x, dif + offset, "-", color="blue", linewidth=1, label="Diferencia")

        # 3) Línea de referencia para la diferencia
        plt.axhline(offset, color="gray", linestyle="--", linewidth=0.5)

        plt.xlabel("2θ (grados)")
        plt.ylabel("Intensidad")
        plt.title("Resultado del Refinamiento Rietveld")
        plt.legend()
        plt.tight_layout()
        plt.show()


# Ahora hay que mostrarle al usuario todas las magnitudes físicas referentes a la muestra.

if __name__ == "__main__":

    # 1) Sacamos las herramientas
    exp_model = ExperimentalModel(
        r"/mnt/c/Users/Manu/Desktop/ICMS/XRD Analysis/NGWO-5EU_10-90(0.02-100)m.xrdml"
    )  # hay que meter aquí la URL o path del xrdml
    theo_model = TheoreticalModel(
        r"/mnt/c/Users/Manu/Desktop/ICMS/cifs/Scheelite.cif"
    )  # hay que meter aquí la URL o path del cif

    # 2) Visualizamos las gráficas iniciales
    exp_model.plot(label="Experimental")
    theo_model.plot(label="Teórico")
    plt.show()

    # 3) Utilizamos las herramientas
    refinement = RietveldSolver(exp_model, theo_model)
    refinement.refine()

    # 4) Empaquetamos el resultado
    sample_cif = theo_model.get_physical_report()

    # 5) Plotamos la curva refinada
    refinement.plot_refinement()

    # 6) Imprimimos los resultados de forma limpia
    print("\n--- RESULTADOS DEL REFINAMIENTO ---")
    for magnitude, value in sample_cif.items():
        print(f"{magnitude} = {value:.4f}")
