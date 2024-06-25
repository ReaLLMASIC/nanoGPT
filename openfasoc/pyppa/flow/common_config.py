from typing import TypedDict, Union, Optional
from os import path

class __FlowConfigDirectories(TypedDict):
	"""The flow directories."""
	FLOW_HOME: str
	"""The home directory for the flow scripts. Default: `.`"""
	WORK_HOME: str
	"""The directory in which all the outputs are generated."""
	DESIGN_HOME: str
	"""The directory in which all design files are stored. Default: `[flow_home]/designs/`"""
	DESIGN_DIR: str
	"""The directory in which the design files for the used design are stored. Default: `[design_home]/[platform]/[design_name]/"""
	UTILS_DIR: str
	"""The directory in which util functions are stored. Default: `[flow_home]/util/`"""
	SCRIPTS_DIR: str
	"""The directory in which the scripts are stored. Default: `[flow_home]/scripts/`"""
	PLATFORM_HOME: str
	"""The directory in which all platform configurations are stored. Default: `[flow_home]/platforms/`"""
	PLATFORM_DIR: str
	"""The directory in which the configuration for the used platform configurations are stored. Default: `[platform_home]/[platform]`"""
	RESULTS_DIR: str
	"""The directory in which all flow results will be generated. Default: `[flow_home]/results/[platform]/[design_name]/`"""
	LOG_DIR: str
	"""The directory in which all log files will be generated. Default: `[flow_home]/logs/[platform]/[design_name]/`"""
	REPORTS_DIR: str
	"""The directory in which all reports will be generated. Default: `[flow_home]/reports/[platform]/[design_name]/`"""
	OBJECTS_DIR: str
	"""The directory in which all objects will be generated. Default: `[flow_home]/objects/[platform]/[design_name]/`"""

class __FlowConfigTools(TypedDict):
	"""The tool configurations."""
	# TOOL COMMANDS
	YOSYS_CMD: str
	"""Path to the the Yosys executable. Default: `/usr/bin/yosys`"""
	OPENROAD_EXE: str
	"""Path to the OpenROAD executable. Default: `/usr/bin/openroad`"""
	KLAYOUT_CMD: str
	"""Path to the Klayout executable. Default: `/usr/bin/klayout`"""

	# SYNTHESIS CONFIG
	SYNTH_ARGS: str
	SYNTH_HIERARCHICAL: str
	"""Optional synthesis variables for Yosys."""

FlowCommonConfigDict = Union[__FlowConfigDirectories, __FlowConfigTools]

FLOW_COMMON_CONFIG_DEFAULTS: FlowCommonConfigDict = {
	'FLOW_HOME': path.abspath('.'),
	'WORK_HOME': path.abspath('.'),
	'YOSYS_CMD': '/usr/bin/yosys',
	'OPENROAD_EXE': '/usr/bin/openroad',
	'KLAYOUT_CMD': '/usr/bin/klayout',
	'SYNTH_ARGS': '-flatten'
}

class FlowCommonConfig:
	configopts: Union[FlowCommonConfigDict, dict]
	config: FlowCommonConfigDict

	def __init__(self):
		# self.configopts = configopts.copy()
		self.config = {**FLOW_COMMON_CONFIG_DEFAULTS, **self.config}

		self.calculate_dirs()

	def calculate_dirs(self):
		# Set defaults for static directories
		self.config['DESIGN_HOME'] = self.configopts.get('DESIGN_HOME', path.join(self.config['FLOW_HOME'], 'designs'))
		self.config['DESIGN_DIR'] = self.configopts.get('DESIGN_DIR', path.join(self.config['DESIGN_HOME'], self.config['PLATFORM'], self.config['DESIGN_NAME']))
		self.config['UTILS_DIR'] = self.configopts.get('UTILS_DIR', path.join(self.config['FLOW_HOME'], 'util'))
		self.config['SCRIPTS_DIR'] = self.configopts.get('SCRIPTS_DIR', path.join(self.config['FLOW_HOME'], 'scripts'))
		self.config['PLATFORM_HOME'] = self.configopts.get('PLATFORM_HOME', path.join(self.config['FLOW_HOME'], 'platforms'))
		self.config['PLATFORM_DIR'] = self.configopts.get('PLATFORM_DIR', path.join(self.config['PLATFORM_HOME'], self.config['PLATFORM']))

		# Set defaults for generated directories
		self.config['RESULTS_DIR'] = self.configopts.get('RESULTS_DIR', path.join(self.config['WORK_HOME'], 'results'))
		self.config['LOG_DIR'] = self.configopts.get('LOG_DIR', path.join(self.config['WORK_HOME'], 'logs'))
		self.config['REPORTS_DIR'] = self.configopts.get('REPORTS_DIR', path.join(self.config['WORK_HOME'], 'reports'))
		self.config['OBJECTS_DIR'] = self.configopts.get('OBJECTS_DIR', path.join(self.config['WORK_HOME'], 'objects'))

	def get_env(self, init_env: Optional[dict]):
		env = {**init_env} if init_env is not None else {**self.config}

		return env
