# MyShop

#FE:
ng new e-shop-frontend --standalone --skip-tests --routing --style=scss
cd myshop
ng add @ngrx/store@latest
ng add @ngrx/effects@latest
ng add @ngrx/store-devtools@latest
ng add @ngrx/router-store@latest
ng add @ngrx/entity@latest
ng add @ngrx/component-store@latest
ng add @angular/material
npm install tailwindcss@latest postcss@latest autoprefixer@latest
npx tailwindcss init
# Nội dung đã được tạo bởi lệnh init, cần update:
cat > tailwind.config.js << EOL
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  important: false, // Đặt true nếu cần override các style của PrimeNG
  theme: {
    extend: {},
  },
  corePlugins: {
    preflight: false, // Ngăn TailwindCSS reset các style mặc định
  },
  plugins: [],
}
EOL

-----------------------

cat > src/styles.scss << EOL
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Your custom styles below */
EOL
-----------------------
# Cài đặt PrimeNG phiên bản mới nhất tương thích với Angular 19
npm install primeng@latest

# Cài đặt PrimeIcons
npm install primeicons

# Cài đặt PrimeFlex (optional - hệ thống grid và utilities)
npm install primeflex
ng add @angular/pwa
# Thư viện cho forms
npm install @angular/forms

----------------------------
ng config projects.e-shop-frontend.architect.build.options.styles \
  "[\"src/styles.scss\", \"node_modules/primeng/resources/themes/lara-light-blue/theme.css\", \"node_modules/primeng/resources/primeng.min.css\", \"node_modules/primeicons/primeicons.css\", \"node_modules/primeflex/primeflex.css\"]"

Import các module PrimeNG cần thiết trong app.config.ts
cat > src/app/app.config.ts << EOL
import { ApplicationConfig } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideAnimations } from '@angular/platform-browser/animations';
import { provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';

import { routes } from './app.routes';
import { provideStore } from '@ngrx/store';
import { provideEffects } from '@ngrx/effects';

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    provideAnimations(),
    provideHttpClient(withInterceptorsFromDi()),
    provideStore(),
    provideEffects()
  ]
};
EOL

Tạo module riêng cho PrimeNG
mkdir -p src/app/shared/prime-ng

cat > src/app/shared/prime-ng/prime-ng.module.ts << EOL
import { NgModule } from '@angular/core';

// Import các module PrimeNG cần thiết
import { ButtonModule } from 'primeng/button';
import { TableModule } from 'primeng/table';
import { ToastModule } from 'primeng/toast';
import { CalendarModule } from 'primeng/calendar';
import { SliderModule } from 'primeng/slider';
import { MultiSelectModule } from 'primeng/multiselect';
import { ContextMenuModule } from 'primeng/contextmenu';
import { DialogModule } from 'primeng/dialog';
import { DropdownModule } from 'primeng/dropdown';
import { ProgressBarModule } from 'primeng/progressbar';
import { InputTextModule } from 'primeng/inputtext';
import { FileUploadModule } from 'primeng/fileupload';
import { ToolbarModule } from 'primeng/toolbar';
import { RatingModule } from 'primeng/rating';
import { RadioButtonModule } from 'primeng/radiobutton';
import { InputNumberModule } from 'primeng/inputnumber';
import { ConfirmDialogModule } from 'primeng/confirmdialog';
import { ConfirmationService } from 'primeng/api';
import { MessageService } from 'primeng/api';
import { InputTextareaModule } from 'primeng/inputtextarea';
import { CardModule } from 'primeng/card';
import { ChartModule } from 'primeng/chart';
import { MenuModule } from 'primeng/menu';
import { MenubarModule } from 'primeng/menubar';
import { PanelMenuModule } from 'primeng/panelmenu';
import { SidebarModule } from 'primeng/sidebar';

const primeNgModules = [
  ButtonModule,
  TableModule,
  ToastModule,
  CalendarModule,
  SliderModule,
  MultiSelectModule,
  ContextMenuModule,
  DialogModule,
  DropdownModule,
  ProgressBarModule,
  InputTextModule,
  FileUploadModule,
  ToolbarModule,
  RatingModule,
  RadioButtonModule,
  InputNumberModule,
  ConfirmDialogModule,
  InputTextareaModule,
  CardModule,
  ChartModule,
  MenuModule,
  MenubarModule,
  PanelMenuModule,
  SidebarModule
];

@NgModule({
  imports: [...primeNgModules],
  exports: [...primeNgModules],
  providers: [MessageService, ConfirmationService]
})
export class PrimeNgModule { }
EOL
------------------

# RxJS
npm install rxjs@latest

# HTTP Client
npm install @angular/common

# Router
npm install @angular/router

# Thư viện xử lý dates
npm install date-fns

# Utilities
npm install lodash-es
npm install @types/lodash-es --save-dev

# Charts (nếu cần biểu đồ thống kê)
npm install chart.js ng2-charts@latest

# Icons
npm install @angular/material-luxon-adapter

# Component Dev Kit (CDK)
npm install @angular/cdk@latest

# Drag & Drop
npm install @angular/cdk/drag-drop
# ESLint
ng add @angular-eslint/schematics

# Prettier
npm install prettier --save-dev
npm install eslint-config-prettier --save-dev

# Jest (nếu muốn dùng thay cho Jasmine)
npm install jest @types/jest jest-preset-angular --save-dev

----------------------------------

# Core folder
mkdir -p src/app/core/auth src/app/core/http src/app/core/models src/app/core/services src/app/core/store

# Features folder 
mkdir -p src/app/features/auth/components src/app/features/auth/services src/app/features/auth/store src/app/features/auth/models src/app/features/auth/guards
mkdir -p src/app/features/products/components src/app/features/products/services src/app/features/products/store
mkdir -p src/app/features/cart src/app/features/checkout
mkdir -p src/app/features/admin/product-management src/app/features/admin/order-management src/app/features/admin/inventory src/app/features/admin/reports
mkdir -p src/app/features/user-profile

# Shared folder
mkdir -p src/app/shared/components src/app/shared/directives src/app/shared/pipes src/app/shared/utils

# Layout folder
mkdir -p src/app/layout/header src/app/layout/footer src/app/layout/sidebar src/app/layout/main-layout

------------------------------------

# Tạo các file môi trường
mkdir -p src/environments
touch src/environments/environment.ts src/environments/environment.development.ts src/environments/environment.production.ts

# Nội dung environment.ts
cat > src/environments/environment.ts << EOL
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000/api',
};
EOL

# Nội dung environment.production.ts
cat > src/environments/environment.production.ts << EOL
export const environment = {
  production: true,
  apiUrl: 'https://api.your-production-domain.com/api',
};
EOL

-----------------------------------------------------------------------------------------------------------------------------------

#BE
python --version  # Kiểm tra phiên bản (nên dùng Python 3.11+)
curl -sSL https://install.python-poetry.org | python3 -
mkdir -p e-shop-backend
cd e-shop-backend
mkdir -p .github/workflows docker/dev docker/prod services shared/models shared/utils shared/messaging k8s/base k8s/overlays/{dev,staging,production} scripts


# Khởi tạo pyproject.toml
cat > pyproject.toml << EOL
[tool.poetry]
name = "e-shop-backend"
version = "0.1.0"
description = "E-Shop Backend Microservices"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"

[tool.poetry.group.dev.dependencies]
black = "^23.3.0"
isort = "^5.12.0"
mypy = "^1.3.0"
flake8 = "^6.0.0"
pytest = "^7.3.1"
pytest-cov = "^4.1.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
EOL

# Khởi tạo Poetry
poetry install


# Tạo script tạo microservice mới
cat > scripts/create_service.sh << EOL
#!/bin/bash

if [ -z "\$1" ]; then
    echo "Usage: \$0 <service-name>"
    exit 1
fi

SERVICE_NAME=\$1
SERVICE_DIR="services/\${SERVICE_NAME}"

# Create service directory structure
mkdir -p \${SERVICE_DIR}/src/api/v1/endpoints
mkdir -p \${SERVICE_DIR}/src/core
mkdir -p \${SERVICE_DIR}/src/db/repositories
mkdir -p \${SERVICE_DIR}/src/models
mkdir -p \${SERVICE_DIR}/src/schemas
mkdir -p \${SERVICE_DIR}/src/services
mkdir -p \${SERVICE_DIR}/src/events
mkdir -p \${SERVICE_DIR}/tests/api
mkdir -p \${SERVICE_DIR}/tests/services
mkdir -p \${SERVICE_DIR}/alembic/versions

# Create main.py
cat > \${SERVICE_DIR}/src/main.py << EOF
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
def health_check():
    return {"status": "ok"}
EOF

# Create config.py
cat > \${SERVICE_DIR}/src/core/config.py << EOF
from pydantic import BaseSettings, PostgresDsn, validator
from typing import List, Optional, Dict, Any
import secrets


class Settings(BaseSettings):
    PROJECT_NAME: str = "${SERVICE_NAME}"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:4200", "http://localhost:3000"]
    
    # Database
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
EOF

# Create router.py
cat > \${SERVICE_DIR}/src/api/v1/router.py << EOF
from fastapi import APIRouter

api_router = APIRouter()

# Import and include routers from endpoints
# Example: from .endpoints import items, users
# api_router.include_router(items.router, prefix="/items", tags=["items"])
EOF

# Create dependencies.py
cat > \${SERVICE_DIR}/src/dependencies.py << EOF
from typing import Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlalchemy.orm import Session

from db.session import SessionLocal
from core.config import settings
from schemas.token import TokenPayload
from models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=["HS256"]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = db.query(User).filter(User.id == token_data.sub).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user
EOF

# Create session.py
cat > \${SERVICE_DIR}/src/db/session.py << EOF
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from core.config import settings

engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
EOF

# Create base.py
cat > \${SERVICE_DIR}/src/db/base.py << EOF
# Import all the models, so that Base has them before being
# imported by Alembic
from db.session import Base
from models.item import Item  # noqa
# from models.user import User  # noqa
EOF

# Create requirements.txt
cat > \${SERVICE_DIR}/requirements.txt << EOF
fastapi>=0.110.0,<0.111.0
uvicorn>=0.27.0,<0.28.0
sqlalchemy>=2.0.25,<2.1.0
psycopg2-binary>=2.9.9,<3.0.0
alembic>=1.13.1,<1.14.0
pydantic>=2.5.3,<2.6.0
python-jose>=3.3.0,<3.4.0
passlib>=1.7.4,<1.8.0
python-multipart>=0.0.6,<0.1.0
aiokafka>=0.8.1,<0.9.0
pytest>=7.4.0,<7.5.0
httpx>=0.26.0,<0.27.0
tenacity>=8.2.3,<8.3.0
EOF

# Create Dockerfile
cat > \${SERVICE_DIR}/Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create alembic.ini
cat > \${SERVICE_DIR}/alembic.ini << EOF
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = postgresql://postgres:postgres@localhost/app

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
EOF

# Create alembic env.py
cat > \${SERVICE_DIR}/alembic/env.py << EOF
from logging.config import fileConfig
import os
import sys

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add the src directory to the path so we can import our models
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the Base and models
from src.db.session import Base
from src.db.base import Base  # noqa

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=url,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
EOF

# Create .env.example
cat > \${SERVICE_DIR}/.env.example << EOF
# Database
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=${SERVICE_NAME}

# JWT
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=43200  # 30 days

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
EOF

echo "Microservice \${SERVICE_NAME} created successfully!"
EOL

chmod +x scripts/create_service.sh
-------------------------------------------
# Tạo API Gateway service
mkdir -p services/gateway/src/{api,core}

# Tạo main.py cho gateway
cat > services/gateway/src/main.py << EOL
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from core.config import settings
import logging

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client
http_client = httpx.AsyncClient()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.api_route("/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def gateway(service_name: str, path: str, request: Request):
    try:
        # Determine target service URL
        if service_name not in settings.SERVICE_URLS:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Service {service_name} not found"}
            )
        
        target_url = f"{settings.SERVICE_URLS[service_name]}/{path}"
        
        # Get request body if any
        body = await request.body()
        
        # Get request headers
        headers = dict(request.headers)
        # Don't forward Host header
        headers.pop("host", None)
        
        # Forward the request to the target service
        response = await http_client.request(
            request.method,
            target_url,
            content=body,
            headers=headers,
            params=request.query_params,
            follow_redirects=True,
        )
        
        # Return the response from the target service
        content = response.content
        return JSONResponse(
            status_code=response.status_code,
            content=response.json() if content else None,
            headers=dict(response.headers),
        )
    except Exception as e:
        logger.error(f"Gateway error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOL

# Tạo config.py cho gateway
cat > services/gateway/src/core/config.py << EOL
from pydantic import BaseSettings
from typing import Dict, List


class Settings(BaseSettings):
    PROJECT_NAME: str = "API Gateway"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:4200", "http://localhost:3000"]
    
    # Service URLs
    SERVICE_URLS: Dict[str, str] = {
        "users": "http://user-service:8000",
        "products": "http://product-service:8000",
        "orders": "http://order-service:8000",
        "inventory": "http://inventory-service:8000",
        "payments": "http://payment-service:8000",
        "analytics": "http://analytics-service:8000",
    }
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
EOL

# Tạo requirements.txt cho gateway
cat > services/gateway/requirements.txt << EOL
fastapi>=0.110.0,<0.111.0
uvicorn>=0.27.0,<0.28.0
httpx>=0.26.0,<0.27.0
pydantic>=2.5.3,<2.6.0
EOL

# Tạo Dockerfile cho gateway
cat > services/gateway/Dockerfile << EOL
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOL
------------------------------------------------------
# Tạo các microservices
./scripts/create_service.sh user-service
./scripts/create_service.sh product-service
./scripts/create_service.sh order-service
./scripts/create_service.sh inventory-service
./scripts/create_service.sh payment-service
./scripts/create_service.sh analytics-service
------------------------------------------------------
# Tạo shared models
mkdir -p shared/models
cat > shared/models/__init__.py << EOL
# Shared models package
EOL

cat > shared/models/common.py << EOL
from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field


class CommonBase(BaseModel):
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


class PaginationParams(BaseModel):
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    page: int
    page_size: int
    pages: int
EOL

# Tạo shared utils
mkdir -p shared/utils
cat > shared/utils/__init__.py << EOL
# Shared utils package
EOL

cat > shared/utils/security.py << EOL
from datetime import datetime, timedelta
from typing import Any, Optional, Union

from jose import jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    subject: Union[str, Any], 
    secret_key: str,
    expires_delta: Optional[timedelta] = None,
    scopes: list[str] = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode = {"exp": expire, "sub": str(subject)}
    if scopes:
        to_encode["scopes"] = scopes
        
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
EOL

# Tạo shared messaging
mkdir -p shared/messaging
cat > shared/messaging/__init__.py << EOL
# Shared messaging package
EOL

cat > shared/messaging/kafka.py << EOL
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import aiokafka
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

logger = logging.getLogger(__name__)


class KafkaClient:
    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str,
        group_id: Optional[str] = None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.group_id = group_id or f"{client_id}-group"
        self.producer = None
        self.consumer = None
        self.admin_client = None
        
    async def create_topics(self, topics: List[str], partitions: int = 1, replication_factor: int = 1) -> None:
        """Create Kafka topics if they don't exist."""
        self.admin_client = AIOKafkaAdminClient(
            bootstrap_servers=self.bootstrap_servers,
            client_id=f"{self.client_id}-admin"
        )
        
        await self.admin_client.start()
        
        try:
            topic_list = [
                NewTopic(name=topic, num_partitions=partitions, replication_factor=replication_factor)
                for topic in topics
            ]
            await self.admin_client.create_topics(topic_list)
            logger.info(f"Created topics: {topics}")
        except Exception as e:
            logger.error(f"Error creating topics: {e}")
        finally:
            await self.admin_client.close()
            
    async def init_producer(self) -> None:
        """Initialize Kafka producer."""
        if self.producer is None:
            self.producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
            )
            await self.producer.start()
            logger.info("Kafka producer initialized")
    
    async def init_consumer(self, topics: List[str]) -> None:
        """Initialize Kafka consumer."""
        if self.consumer is None:
            self.consumer = aiokafka.AIOKafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                client_id=self.client_id,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
            )
            await self.consumer.start()
            logger.info(f"Kafka consumer initialized for topics: {topics}")
    
    async def send_message(self, topic: str, key: Optional[str], value: Dict[str, Any]) -> None:
        """Send message to Kafka topic."""
        if self.producer is None:
            await self.init_producer()
            
        try:
            key_bytes = key.encode("utf-8") if key else None
            value_bytes = json.dumps(value).encode("utf-8")
            await self.producer.send_and_wait(topic, value=value_bytes, key=key_bytes)
            logger.debug(f"Message sent to {topic}: {value}")
        except Exception as e:
            logger.error(f"Error sending message to {topic}: {e}")
            raise
            
    async def consume_messages(self, message_handler: Callable[[Dict[str, Any], str], None]) -> None:
        """Consume messages from subscribed topics."""
        if self.consumer is None:
            raise ValueError("Consumer not initialized. Call init_consumer first.")
            
        try:
            async for message in self.consumer:
                try:
                    # Parse the message value as JSON
                    value = json.loads(message.value.decode("utf-8"))
                    
                    # Process the message
                    logger.debug(f"Received message from {message.topic}: {value}")
                    await message_handler(value, message.topic)
                    
                    # Commit the offset
                    await self.consumer.commit()
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse message: {message.value}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
            
    async def close(self) -> None:
        """Close Kafka producer and consumer."""
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer closed")
            
        if self.consumer:
            await self.consumer.stop()
            logger.info("Kafka consumer closed")
EOL

-------------------------------------------------------
# Tạo docker-compose.yml cho phát triển
cat > docker/dev/docker-compose.yml << EOL
version: '3.8'

services:
  # API Gateway
  gateway:
    build:
      context: ../../services/gateway
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - CORS_ORIGINS=["http://localhost:4200"]
    depends_on:
      - user-service
      - product-service
      - order-service
      - inventory-service
      - payment-service
      - analytics-service
    networks:
      - backend-network

  # User Service
  user-service:
    build:
      context: ../../services/user-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=user_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Product Service
  product-service:
    build:
      context: ../../services/product-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=product_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Order Service
  order-service:
    build:
      context: ../../services/order-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=order_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Inventory Service
  inventory-service:
    build:
      context: ../../services/inventory-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=inventory_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Payment Service
  payment-service:
    build:
      context: ../../services/payment-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=payment_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Analytics Service
  analytics-service:
    build:
      context: ../../services/analytics-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=analytics_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
      - redis
    networks:
      - backend-network

  # PostgreSQL
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_MULTIPLE_DATABASES=user_db,product_db,order_db,inventory_db,payment_db,analytics_db
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-multiple-dbs.sh:/docker-entrypoint-initdb.d/init-multiple-dbs.sh
    ports:
      - "5432:5432"
    networks:
      - backend-network

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - backend-network

  # Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.3
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - backend-network

  kafka:
    image: confluentinc/cp-kafka:7.4.3
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    networks:
      - backend-network
      
  # Kafka UI
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
      KAFKA_CLUSTERS_0_ZOOKEEPER: zookeeper:2181
    depends_on:
      - kafka
      - zookeeper
    networks:
      - backend-network

  # PGAdmin
  pgadmin:
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@admin.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres
    networks:
      - backend-network

networks:
  backend-network:

volumes:
  postgres-data:
EOL

# Tạo script để tạo nhiều database trong Postgres
cat > docker/dev/init-multiple-dbs.sh << EOL
#!/bin/bash

set -e
set -u

function create_database() {
    local database=\$1
    echo "Creating database '\$database'"
    psql -v ON_ERROR_STOP=1 --username "\$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE \$database;
        GRANT ALL PRIVILEGES ON DATABASE \$database TO \$POSTGRES_USER;
EOSQL
}

if [ -n "\$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "Creating multiple databases: \$POSTGRES_MULTIPLE_DATABASES"
    for db in \$(echo \$POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
        create_database \$db
    done
    echo "Multiple databases created"
fi
EOL

chmod +x docker/dev/init-multiple-dbs.sh

--------------------------------------------

# Tạo docker-compose.yml cho phát triển
cat > docker/dev/docker-compose.yml << EOL
version: '3.8'

services:
  # API Gateway
  gateway:
    build:
      context: ../../services/gateway
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - CORS_ORIGINS=["http://localhost:4200"]
    depends_on:
      - user-service
      - product-service
      - order-service
      - inventory-service
      - payment-service
      - analytics-service
    networks:
      - backend-network

  # User Service
  user-service:
    build:
      context: ../../services/user-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=user_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Product Service
  product-service:
    build:
      context: ../../services/product-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=product_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Order Service
  order-service:
    build:
      context: ../../services/order-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=order_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Inventory Service
  inventory-service:
    build:
      context: ../../services/inventory-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=inventory_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Payment Service
  payment-service:
    build:
      context: ../../services/payment-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=payment_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
    networks:
      - backend-network

  # Analytics Service
  analytics-service:
    build:
      context: ../../services/analytics-service
      dockerfile: Dockerfile
    environment:
      - POSTGRES_SERVER=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=analytics_db
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - postgres
      - kafka
      - redis
    networks:
      - backend-network

  # PostgreSQL
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_MULTIPLE_DATABASES=user_db,product_db,order_db,inventory_db,payment_db,analytics_db
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-multiple-dbs.sh:/docker-entrypoint-initdb.d/init-multiple-dbs.sh
    ports:
      - "5432:5432"
    networks:
      - backend-network

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - backend-network

  # Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.3
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - backend-network

  kafka:
    image: confluentinc/cp-kafka:7.4.3
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    networks:
      - backend-network
      
  # Kafka UI
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
      KAFKA_CLUSTERS_0_ZOOKEEPER: zookeeper:2181
    depends_on:
      - kafka
      - zookeeper
    networks:
      - backend-network

  # PGAdmin
  pgadmin:
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@admin.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres
    networks:
      - backend-network

networks:
  backend-network:

volumes:
  postgres-data:
EOL

# Tạo script để tạo nhiều database trong Postgres
cat > docker/dev/init-multiple-dbs.sh << EOL
#!/bin/bash

set -e
set -u

function create_database() {
    local database=\$1
    echo "Creating database '\$database'"
    psql -v ON_ERROR_STOP=1 --username "\$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE \$database;
        GRANT ALL PRIVILEGES ON DATABASE \$database TO \$POSTGRES_USER;
EOSQL
}

if [ -n "\$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "Creating multiple databases: \$POSTGRES_MULTIPLE_DATABASES"
    for db in \$(echo \$POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
        create_database \$db
    done
    echo "Multiple databases created"
fi
EOL

chmod +x docker/dev/init-multiple-dbs.sh

---------------------------------

# Tạo Kubernetes base configs
mkdir -p k8s/base/{gateway,user-service,product-service,order-service,inventory-service,payment-service,analytics-service}

# Tạo kustomization.yaml cho base
cat > k8s/base/kustomization.yaml << EOL
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - gateway
  - user-service
  - product-service
  - order-service
  - inventory-service
  - payment-service
  - analytics-service
EOL

# Tạo .env.example tổng
cat > .env.example << EOL
# Common settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
SECRET_KEY=your-secret-key

# Service specific settings
USER_DB=user_db
PRODUCT_DB=product_db
ORDER_DB=order_db
INVENTORY_DB=inventory_db
PAYMENT_DB=payment_db
ANALYTICS_DB=analytics_db

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
EOL

# Tạo script để chạy các migrations
cat > scripts/run_migrations.py << EOL
#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

SERVICES = [
    "user-service",
    "product-service",
    "order-service",
    "inventory-service",
    "payment-service",
    "analytics-service"
]

def run_migrations(service_name: str) -> None:
    """Run Alembic migrations for a service."""
    service_dir = Path(f"services/{service_name}")
    if not service_dir.exists():
        print(f"Service directory {service_dir} does not exist.")
        return
        
    alembic_dir = service_dir / "alembic"
    if not alembic_dir.exists():
        print(f"Alembic directory {alembic_dir} does not exist.")
        return
        
    print(f"Running migrations for {service_name}...")
    os.chdir(service_dir)
    
    try:
        # Run alembic upgrade
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        print(f"Successfully ran migrations for {service_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run migrations for {service_name}: {e}")
    finally:
        os.chdir("../..")

def main() -> None:
    """Main function to run migrations."""
    if len(sys.argv) > 1:
        service_name = sys.argv[1]
        if service_name not in SERVICES:
            print(f"Unknown service: {service_name}")
            print(f"Available services: {', '.join(SERVICES)}")
            sys.exit(1)
        run_migrations(service_name)
    else:
        for service in SERVICES:
            run_migrations(service)

if __name__ == "__main__":
    main()
EOL

chmod +x scripts/run_migrations.py

# Tạo script để chạy dự án
cat > scripts/start_dev.sh << EOL
#!/bin/bash

# Change to the docker/dev directory
cd docker/dev

# Start the services with docker-compose
docker-compose up -d

echo "Services started. API Gateway is available at http://localhost:8000"
echo "Kafka UI is available at http://localhost:8080"
echo "PGAdmin is available at http://localhost:5050"
echo "Username: admin@admin.com, Password: admin"
EOL

chmod +x scripts/start_dev.sh

-----------------------
# Khởi động môi trường dev
./scripts/start_dev.sh

# Chạy migrations
python scripts/run_migrations.py



