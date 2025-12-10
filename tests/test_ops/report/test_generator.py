from src.ops.report.generator import ReportGenerator

def test_report_generator_init():
    gen = ReportGenerator(experiments_base_dir='artifacts/experiments', logger=None)
    assert hasattr(gen, 'experiments_base_dir')
