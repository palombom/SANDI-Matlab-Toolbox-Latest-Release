% HTMLREPORTDEMO
% Generates a demo report in the Output directory of this module
%
%
clear
r = report_generator('Test Report', 'Output');
r.open();
r.section('A Demo Report');
r.add_text('This is an example HTML report generated from Matlab. Discuss what you did here, maybe give a few example calculations:');
r.add_text(sprintf('exp(-2*pi*0.5) = %f', exp(-2*pi*0.5)));
r.subsection('Figures')
r.add_text('This is a subsection demonstrating how figures/plots can be added directly to the report');
logo;
r.add_figure(gcf,'You can add figures to the report directly','left');
r.add_figure(gcf,'And align them however you want', 'centered');
r.end_section(); 
r.end_section(); 
r.close();