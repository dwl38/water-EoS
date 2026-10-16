"""
Important numerical constants.
"""

K_B = 8.617333262E-5
"""Boltzmann constant, in eV/K."""

AVOGADRO_NUMBER = 6.02214076E23
"""Avogadro number."""

WATER_MOLAR_MASS = 18.015
"""Water molar mass, in g/mol."""

WATER_MASS_DENSITY_CONVERSION = 1.0E27 * WATER_MOLAR_MASS / AVOGADRO_NUMBER
"""Conversion ratio from number density, in Å⁻³, to water mass density, in
kg/m³."""

WATER_NUMBER_DENSITY_CONVERSION = 1.0E-27 * AVOGADRO_NUMBER / WATER_MOLAR_MASS
"""Conversion ratio from water mass density, in kg/m³, to number density, in
Å⁻³."""

VISCOSITY_CONVERSION = 1.0E-4
"""Conversion ratio from bar⋅ps to mPa⋅s."""

PRESSURE_CONVERSION = 1602176.634
"""Conversion ratio from eV/Å³ to bar."""

CRITICAL_VOLUME = 92.9039635
"""Water's critical point volume, in Å³."""

CRITICAL_TEMPERATURE = 647.096
"""Water's critical point temperature, in K."""

AMBIENT_VOLUME = 30.003
"""Water's volume at rtp conditions, in Å³."""

AMBIENT_TEMPERATURE = 298.15
"""Temperature corresponding to rtp conditions, in K."""
