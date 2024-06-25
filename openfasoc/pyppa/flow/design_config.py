from typing import TypedDict, Union, Optional
from os import path

class __DesignCommonConfig(TypedDict):
	"""The common design configuration."""
	DESIGN_NAME: str
	"""The name of the design."""
	PLATFORM: str
	"""The process design kit to be used."""
	VERILOG_FILES: list[str]
	"""The paths to the design Verilog files."""
	SDC_FILE: str
	"""The path to design constraint (SDC) file. Default: `[design_dir]/constraint.sdc"""
	ABC_AREA: bool
	"""Whether to use `ABC_AREA` strategy for Yosys synthesis. Setting it to false will use `ABC_SPEED` strategy. Default: `False`"""
	ABC_CLOCK_PERIOD_IN_PS: float
	"""Clock period to be used by STA during synthesis. Default value read from `constraint.sdc`."""

class __DesignSynthConfig(TypedDict):
	"""The synthesis design configuration."""
	PRESERVE_CELLS: list[str]
	"""The list of cells to preserve the hierarchy of during synthesis."""
	PRESYNTH_VCD: str
	"""Path to a presynthesis simulation VCD file used for static timing analysis (STA)."""

class __DesignFloorplanConfig(TypedDict):
	"""The floorplan design configuration."""
	FLOORPLAN_DEF: str
	"""Use the DEF file to initialize floorplan."""
	DIE_AREA: tuple[float, float, float, float]
	"""The die area specified as a tuple of lower-left and upper-right corners in microns (X1,Y1,X2,Y2). This variable is ignored if `CORE_UTILIZATION` and `CORE_ASPECT_RATIO` are defined."""
	CORE_AREA: tuple[float, float, float, float]
	"""The core area specified as a tuple of lower-left and upper-right corners in microns (X1,Y1,X2,Y2). This variable is ignored if `CORE_UTILIZATION` and `CORE_ASPECT_RATIO` are defined."""
	CORE_UTILIZATION: float
	"""The core utilization percentage (0-100). Overrides `DIE_AREA` and `CORE_AREA`."""
	CORE_ASPECT_RATIO: float
	"""The core aspect ratio (height / width). This values is ignored if `CORE_UTILIZATION` undefined."""
	CORE_MARGIN: int
	"""The margin between the core area and die area, in multiples of SITE heights. The margin is applied to each side. This variable is ignored if `CORE_UTILIZATION` is undefined."""
	PLACE_PINS_ARGS: str
	"""Arguments for io pin placement."""

FlowDesignConfigDict = Union[__DesignCommonConfig, __DesignSynthConfig, __DesignFloorplanConfig]

FLOW_DESIGN_CONFIG_DEFAULTS: FlowDesignConfigDict = {
	'ABC_AREA': False,
	'ABC_CLOCK_PERIOD_IN_PS': 0,
	'PLACE_PINS_ARGS': ''
}

class FlowDesignConfig:
	configopts: Union[FlowDesignConfigDict, dict]
	config: FlowDesignConfigDict

	def __init__(self):
		# self.configopts = configopts.copy()
		self.config = {**FLOW_DESIGN_CONFIG_DEFAULTS, **self.config}

		self.config['SDC_FILE'] = self.config.get('SDC_FILE', path.join(self.config['DESIGN_DIR'], 'constraint.sdc'))

	def get_env(self, init_env: Optional[dict]):
		env = {**init_env} if init_env is not None else {**self.config}

		for key in ('VERILOG_FILES', 'PRESERVE_CELLS', 'DIE_AREA', 'CORE_AREA'):
			if key in self.config:
				env[key] = ' '.join(self.config[key])

		for key in ('CORE_UTILIZATION', 'CORE_ASPECT_RATIO', 'CORE_MARGIN', 'ABC_CLOCK_PERIOD_IN_PS'):
			if key in self.config:
				env[key] = str(self.config[key])

		env['ABC_AREA'] = str(int(self.config['ABC_AREA']))

		return env