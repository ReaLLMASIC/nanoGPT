proc write_power_report {step} {
	set clks [all_clocks]

	if { [llength $clks] == 0 } {
		utl::warn "FLW" 6 "No clocks found."
	} else {
		set_propagated_clock [all_clocks]


		if {[info exists ::env(PRESYNTH_VCD)]} {
			puts "Reading VCD file for setting power activity."
			sta::read_power_activities -vcd $::env(PRESYNTH_VCD)
		} else {
			puts "No VCD file found. Using default power activity values."
			sta::set_power_activity -input -activity .1
			sta::set_power_activity -input_port rst -activity 0
		}

		set power_report_file [file join $::env(REPORTS_DIR) "${step}_power_report.txt"]
		sta::report_power > $power_report_file

		unset_propagated_clock [all_clocks]
	}
}