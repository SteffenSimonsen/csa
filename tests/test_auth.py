"""
Tests for authentication functionality
"""
import pytest
from db.operations import (
    create_user, authenticate_user, hash_password, verify_password,
    create_user_session, convert_anonymous_session_to_user, create_anonymous_session
)
from db.schemas import UserCreate
from db.connection import create_tables
from uuid import uuid4


class TestPasswordHashing:
    """Test password hashing and verification"""

    def test_hash_password(self):
        """Test that password hashing works"""
        password = "TestPassword123"
        hashed = hash_password(password)

        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")  # bcrypt prefix

    def test_verify_password_correct(self):
        """Test password verification with correct password"""
        password = "TestPassword123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password"""
        password = "TestPassword123"
        hashed = hash_password(password)

        assert verify_password("WrongPassword", hashed) is False


class TestUserCreation:
    """Test user creation and retrieval"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure tables exist before tests"""
        create_tables()

    def test_create_user_success(self):
        """Test creating a new user"""
        user_data = UserCreate(
            email=f"test{uuid4()}@example.com",
            username=f"testuser{uuid4().hex[:8]}",
            password="TestPass123"
        )

        user = create_user(user_data)

        assert user.email == user_data.email
        assert user.username == user_data.username.lower()
        assert user.user_id is not None

    def test_create_user_duplicate_email(self):
        """Test that duplicate email raises error"""
        email = f"duplicate{uuid4()}@example.com"
        username1 = f"user1{uuid4().hex[:8]}"
        username2 = f"user2{uuid4().hex[:8]}"

        user_data1 = UserCreate(
            email=email,
            username=username1,
            password="TestPass123"
        )
        create_user(user_data1)

        user_data2 = UserCreate(
            email=email,  # Same email
            username=username2,  # Different username
            password="TestPass456"
        )

        with pytest.raises(ValueError, match="Email .* already exists"):
            create_user(user_data2)

    def test_create_user_duplicate_username(self):
        """Test that duplicate username raises error"""
        username = f"duplicate{uuid4().hex[:8]}"
        email1 = f"email1{uuid4()}@example.com"
        email2 = f"email2{uuid4()}@example.com"

        user_data1 = UserCreate(
            email=email1,
            username=username,
            password="TestPass123"
        )
        create_user(user_data1)

        user_data2 = UserCreate(
            email=email2,  # Different email
            username=username,  # Same username
            password="TestPass456"
        )

        with pytest.raises(ValueError, match="Username .* already exists"):
            create_user(user_data2)


class TestAuthentication:
    """Test user authentication"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure tables exist before tests"""
        create_tables()

    def test_authenticate_user_success(self):
        """Test successful authentication"""
        user_data = UserCreate(
            email=f"auth{uuid4()}@example.com",
            username=f"authuser{uuid4().hex[:8]}",
            password="TestPass123"
        )
        created_user = create_user(user_data)

        # Authenticate with username
        authenticated = authenticate_user(user_data.username, "TestPass123")
        assert authenticated is not None
        assert authenticated.user_id == created_user.user_id

    def test_authenticate_user_with_email(self):
        """Test authentication using email instead of username"""
        user_data = UserCreate(
            email=f"authmail{uuid4()}@example.com",
            username=f"authuser{uuid4().hex[:8]}",
            password="TestPass123"
        )
        created_user = create_user(user_data)

        # Authenticate with email
        authenticated = authenticate_user(user_data.email, "TestPass123")
        assert authenticated is not None
        assert authenticated.user_id == created_user.user_id

    def test_authenticate_user_wrong_password(self):
        """Test authentication with wrong password"""
        user_data = UserCreate(
            email=f"wrongpass{uuid4()}@example.com",
            username=f"wronguser{uuid4().hex[:8]}",
            password="TestPass123"
        )
        create_user(user_data)

        authenticated = authenticate_user(user_data.username, "WrongPassword")
        assert authenticated is None

    def test_authenticate_nonexistent_user(self):
        """Test authentication with nonexistent user"""
        authenticated = authenticate_user("nonexistent", "password")
        assert authenticated is None


class TestSessionConversion:
    """Test converting anonymous sessions to user sessions"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure tables exist before tests"""
        create_tables()

    def test_convert_anonymous_session_to_user(self):
        """Test converting an anonymous session to a user session"""
        # Create a user
        user_data = UserCreate(
            email=f"session{uuid4()}@example.com",
            username=f"sessionuser{uuid4().hex[:8]}",
            password="TestPass123"
        )
        user = create_user(user_data)

        # Create anonymous session
        anon_session = create_anonymous_session()
        assert anon_session.is_anonymous is True
        assert anon_session.user_id is None

        # Convert to user session
        converted = convert_anonymous_session_to_user(anon_session.session_id, user.user_id)

        assert converted.session_id == anon_session.session_id
        assert converted.user_id == user.user_id
        assert converted.is_anonymous is False

    def test_convert_invalid_session(self):
        """Test converting a nonexistent session"""
        fake_session_id = uuid4()
        fake_user_id = uuid4()

        with pytest.raises(ValueError, match="Session not found"):
            convert_anonymous_session_to_user(fake_session_id, fake_user_id)