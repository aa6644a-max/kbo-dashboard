import sys
import os

# PythonAnywhere: 프로젝트 경로를 sys.path에 추가
# 실제 경로로 수정 필요: /home/YOUR_USERNAME/kbo_dashborad
project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from kbo_dashboard import app as application
