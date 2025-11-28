from pathlib import Path
path = Path('capital_programme_optimiser/config/loader.py')
text = path.read_text()
if 'import os' not in text.partition('\n')[0]:
    text = text.replace('from pathlib import Path\nfrom typing import Any, Dict, Iterable, List, Optional\n\nimport yaml', 'from pathlib import Path\nfrom typing import Any, Dict, Iterable, List, Optional\n\nimport os\nimport yaml')
path.write_text(text)
