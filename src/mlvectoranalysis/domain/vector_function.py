from __future__ import annotations

from sympy import Matrix, sqrt, sympify

from mlvectoranalysis.domain.component_function import ComponentFunction


class VectorFunction:
    def __init__(
        self,
        component_x: ComponentFunction,
        component_y: ComponentFunction,
        component_z: ComponentFunction,
        name: str = "r",
        variable: str = "t",
    ) -> None:
        self.name = name
        self.variable = variable
        self.component_x = component_x
        self.component_y = component_y
        self.component_z = component_z

    def get_components(self) -> tuple[ComponentFunction, ComponentFunction, ComponentFunction]:
        return (self.component_x, self.component_y, self.component_z)

    def simplify_all(self) -> "VectorFunction":
        return VectorFunction(
            self.component_x.simplify(),
            self.component_y.simplify(),
            self.component_z.simplify(),
            self.name,
            self.variable,
        )

    def differentiate(self) -> "VectorFunction":
        return VectorFunction(
            ComponentFunction(str(self.component_x.differentiate()), self.variable),
            ComponentFunction(str(self.component_y.differentiate()), self.variable),
            ComponentFunction(str(self.component_z.differentiate()), self.variable),
            self.name,
            self.variable,
        )

    def velocity(self) -> "VectorFunction":
        return self.differentiate()

    def acceleration(self) -> "VectorFunction":
        return self.differentiate().differentiate()

    def evaluate(self, value: float) -> tuple[float, float, float]:
        return (
            self.component_x.evaluate(value),
            self.component_y.evaluate(value),
            self.component_z.evaluate(value),
        )

    def to_matrix(self) -> Matrix:
        return Matrix([
            self.component_x.symbolic_form,
            self.component_y.symbolic_form,
            self.component_z.symbolic_form,
        ])

    def norm(self):
        vector = self.to_matrix()
        return sqrt(vector.dot(vector))

    def speed(self):
        velocity_vector = self.velocity().to_matrix()
        return sqrt(velocity_vector.dot(velocity_vector))
