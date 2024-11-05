import random
import math


class Cell:
    """Superclass of the different types of cells in the model."""

    def __init__(self, stage,
                 average_healthy_glucose_absorption,
                 average_cancer_glucose_absorption,
                 average_healthy_oxygen_consumption,
                 average_cancer_oxygen_consumption,
                 critical_glucose_level,
                 critical_oxygen_level,
                 quiescent_oxygen_level,
                 quiescent_glucose_level,
                 alpha_norm_tissue, beta_norm_tissue,
                 alpha_tumor, beta_tumor):
        
        """Constructor of Cell."""
        self.age = 0
        self.stage = stage
        self.alive = True
        self.efficiency = 0
        self.oxy_efficiency = 0
        self.repair = 0
        self.next_dose = 0
        self.cumulative_dose = 0
        
        self.average_healthy_glucose_absorption = average_healthy_glucose_absorption
        self.average_cancer_glucose_absorption  = average_cancer_glucose_absorption
        
        self.average_healthy_oxygen_consumption = average_healthy_oxygen_consumption
        self.average_cancer_oxygen_consumption  = average_cancer_oxygen_consumption
        
        self.quiescent_glucose_level = quiescent_glucose_level
        self.quiescent_oxygen_level = quiescent_oxygen_level
        
        self.critical_glucose_level = critical_glucose_level
        self.critical_oxygen_level = critical_oxygen_level
        
        self.critical_neighbors = 9
        self.alpha_tumor = alpha_tumor
        self.beta_tumor = beta_tumor
        self.alpha_norm_tissue = alpha_norm_tissue
        self.beta_norm_tissue = beta_norm_tissue
        self.repair_time = 9
        
        self.radiosensitivities = [1, .75, 1.25, 1.25, .75]

    def __lt__(self, other):
        """Used to allow sorting of Cell lists"""
        return -self.cell_type() < -other.cell_type()


class HealthyCell(Cell):
    """HealthyCells are cells representing healthy tissue in the model."""
    cell_count = 0

    def __init__(self, stage,
                 average_healthy_glucose_absorption,
                 average_cancer_glucose_absorption,
                 average_healthy_oxygen_consumption,
                 average_cancer_oxygen_consumption,
                 critical_glucose_level,
                 critical_oxygen_level,
                 quiescent_oxygen_level,
                 quiescent_glucose_level,
                 alpha_norm_tissue, beta_norm_tissue,
                 alpha_tumor, beta_tumor):
        
        """Constructor of a HealthyCell."""
        Cell.__init__(self, stage,
                 average_healthy_glucose_absorption,
                 average_cancer_glucose_absorption,
                 average_healthy_oxygen_consumption,
                 average_cancer_oxygen_consumption,
                 critical_glucose_level,
                 critical_oxygen_level,
                 quiescent_oxygen_level,
                 quiescent_glucose_level,
                 alpha_norm_tissue, beta_norm_tissue,
                 alpha_tumor, beta_tumor)
        
        HealthyCell.cell_count += 1
        factor = random.normalvariate(1, 1/3)
        factor = max(0, min(2, factor))
        self.efficiency = self.average_healthy_glucose_absorption * factor
        self.oxy_efficiency = self.average_healthy_oxygen_consumption * factor

    def cycle(self, glucose, count, oxygen):
        """Simulate an hour of the cell cycle."""
        if glucose < self.critical_glucose_level or oxygen < self.critical_oxygen_level:
            self.alive = False
            HealthyCell.cell_count -= 1
            return 0, 0
        if self.repair == 0:
            self.age += 1
        else:
            self.repair -= 1
        if self.stage == 4:  # Quiescent
            if glucose > self.quiescent_glucose_level and count < self.critical_neighbors and oxygen > self.quiescent_oxygen_level:
                self.age = 0
                self.stage = 0
            return self.efficiency * .75, self.oxy_efficiency * .75
        elif self.stage == 3:  # Mitosis
            if self.age == 1:
                self.stage = 0
                self.age = 0
            return self.efficiency, self.oxy_efficiency, 0
        elif self.stage == 2:  # Gap 2
            if self.age == 4:
                self.age = 0
                self.stage = 3
            return self.efficiency, self.oxy_efficiency
        elif self.stage == 1:  # Synthesis
            if self.age == 8:
                self.age = 0
                self.stage = 2
            return self.efficiency, self.oxy_efficiency
        elif self.stage == 0:  # Gap 1
            if glucose < self.quiescent_glucose_level or count > self.critical_neighbors or oxygen < self.quiescent_oxygen_level:
                self.age = 0
                self.stage = 4
            elif self.age == 11:
                    self.age = 0
                    self.stage = 1
            return self.efficiency, self.oxy_efficiency

    def radiate(self, dose):
        """Irradiate this cell with a specific dose"""
        self.cumulative_dose += dose
        survival_probability = math.exp(self.radiosensitivities[self.stage] * (-self.alpha_norm_tissue*dose - self.beta_norm_tissue * (dose ** 2)))
        if random.uniform(0, 1) > survival_probability:
            self.alive = False
            HealthyCell.cell_count -= 1
        elif dose > 0.5:
            self.repair += int(round(random.uniform(0, 2) * self.repair_time))

    def cell_color(self):
        """RGB for the cell's color"""
        return 0, 204, 102

    def cell_type(self):
        """Return 1, the type of the cell to sort cell lists and compare them"""
        return 1


class CancerCell(Cell):
    """CancerCells are cells representing tumoral tissue in the model."""
    cell_count = 0

    def __init__(self, stage,
                 average_healthy_glucose_absorption,
                 average_cancer_glucose_absorption,
                 average_healthy_oxygen_consumption,
                 average_cancer_oxygen_consumption,
                 critical_glucose_level,
                 critical_oxygen_level,
                 quiescent_oxygen_level,
                 quiescent_glucose_level,
                 cell_cycle,
                 radiosensitivities,
                 alpha_norm_tissue, beta_norm_tissue,
                 alpha_tumor, beta_tumor):
        
        """Constructor of CancerCell."""
        Cell.__init__(self, stage,
                 average_healthy_glucose_absorption,
                 average_cancer_glucose_absorption,
                 average_healthy_oxygen_consumption,
                 average_cancer_oxygen_consumption,
                 critical_glucose_level,
                 critical_oxygen_level,
                 quiescent_oxygen_level,
                 quiescent_glucose_level,
                 alpha_norm_tissue, beta_norm_tissue,
                 alpha_tumor, beta_tumor)
        
        CancerCell.cell_count += 1
        self.cell_cycle = cell_cycle
        self.radiosensitivities=radiosensitivities

    def radiate(self, dose):
        """Irradiate this cell with a specific dose."""
        survival_probability = math.exp(self.radiosensitivities[self.stage] * (-self.alpha_tumor*dose - self.beta_tumor * (dose ** 2)))
        if random.random() > survival_probability:
            self.alive = False
            CancerCell.cell_count -= 1
        elif dose > 0.5:
            self.repair += int(round(random.uniform(0, 2) * self.repair_time))

    def cycle(self, glucose, count, oxygen):
        """Simulate one hour of the cell's cycle"""
        if glucose < self.critical_glucose_level or oxygen < self.critical_oxygen_level:
            self.alive = False
            CancerCell.cell_count -= 1
            return 0, 0
        factor = random.normalvariate(1, 1 / 3)
        factor = max(0, min(2, factor))
        self.efficiency = self.average_cancer_glucose_absorption * factor
        self.oxy_efficiency = self.average_cancer_oxygen_consumption * factor
        if self.repair == 0:
            self.age += 1
        else:
            self.repair -= 1
        if self.stage == 3:  # Mitosis
            if self.age == self.cell_cycle[3]:
                self.stage = 0
                self.age = 0
                return self.efficiency, self.oxy_efficiency, 1
            return self.efficiency, self.oxy_efficiency
        elif self.stage == 2:  # Gap 2
            if self.age == self.cell_cycle[2]:
                self.age = 0
                self.stage = 3
            return self.efficiency, self.oxy_efficiency
        elif self.stage == 1:  # Synthesis
            if self.age == self.cell_cycle[1]:
                self.age = 0
                self.stage = 2
            return self.efficiency, self.oxy_efficiency
        elif self.stage == 0:  # Gap 1
            if self.age == self.cell_cycle[0]:
                self.age = 0
                self.stage = 1
            return self.efficiency, self.oxy_efficiency

    def cell_color(self):
        """RGB for the cell's color"""
        return 104, 24, 24

    def cell_type(self):
        """Return -1, the type of the cell to sort cell lists and compare them"""
        return -1


class OARCell(Cell):
    """OARCells are cells representing an organ at risk in the model."""
    cell_count = 0
    worth = 5

    def __init__(self, stage, worth):
        """Constructor of OARCell"""
        OARCell.cell_count += 1
        Cell.__init__(self, stage)
        OARCell.worth = worth

    def cycle(self, glucose, count, oxygen):
        """Simulate one hour of the cell's cycle"""
        self.age += 1
        if glucose < self.critical_glucose_level or oxygen < self.critical_oxygen_level:
            self.alive = False
            OARCell.cell_count -= 1
            return 0, 0, 2
        elif self.stage == 4:  # Quiescent
            return self.efficiency * .75, self.oxy_efficiency * .75
        elif self.stage == 3:  # Mitosis
            self.stage = 0
            self.age = 0
            return self.efficiency, self.oxy_efficiency, 3
        elif self.stage == 2:  # Gap 2
            if self.age == 4:
                self.age = 0
                self.stage = 3
            return self.efficiency, self.oxy_efficiency
        elif self.stage == 1:  # Synthesis
            if self.age == 8:
                self.age = 0
                self.stage = 2
            return self.efficiency, self.oxy_efficiency
        elif self.stage == 0:  # Gap 1
            if glucose < self.quiescent_glucose_level or count > self.critical_neighbors or oxygen < self.quiescent_oxygen_level:
                self.age = 0
                self.stage = 4
            elif self.age == 11:
                self.age = 0
                self.stage = 1
            return self.efficiency, self.oxy_efficiency

    def cell_color(self):
        """RGB for the cell's color"""
        return 255, 255, 153

    def cell_type(self):
        """Return the OARCell's worth, the type of the cell to sort cell lists and compare them"""
        return OARCell.worth

    def radiate(self, dose):
        """Irradiate this cell with a specific dose."""
        survival_probability = math.exp(self.radiosensitivities[self.stage] * (-self.alpha_norm_tissue*dose - self.beta_norm_tissue * (dose ** 2)))
        if random.random() > survival_probability:
            self.alive = False
            OARCell.cell_count -= 1
