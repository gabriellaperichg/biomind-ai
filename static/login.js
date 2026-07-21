"use strict";

const loginView = document.getElementById("login-view");
const changePasswordView = document.getElementById("change-password-view");

const loginForm = document.getElementById("login-form");
const loginButton = document.getElementById("login-button");
const loginMessage = document.getElementById("login-message");

const emailInput = document.getElementById("email");
const passwordInput = document.getElementById("password");

const changePasswordForm = document.getElementById(
  "change-password-form"
);
const changePasswordButton = document.getElementById(
  "change-password-button"
);
const changePasswordMessage = document.getElementById(
  "change-password-message"
);

const currentPasswordInput = document.getElementById(
  "current-password"
);
const newPasswordInput = document.getElementById("new-password");
const confirmPasswordInput = document.getElementById(
  "confirm-password"
);

let temporaryPassword = "";


function setMessage(element, message = "", type = "error") {
  element.textContent = message;
  element.className = "form-message";

  if (message) {
    element.classList.add(type);
  }
}


function setButtonLoading(button, loading) {
  button.disabled = loading;

  const label = button.querySelector(".button-label");
  const loadingLabel = button.querySelector(".button-loading");

  if (label) {
    label.hidden = loading;
  }

  if (loadingLabel) {
    loadingLabel.hidden = !loading;
  }
}


async function readJsonSafely(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}


async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: {
      ...(options.body
        ? { "Content-Type": "application/json" }
        : {}),
      ...(options.headers || {}),
    },
  });

  const data = await readJsonSafely(response);

  if (!response.ok) {
    const message =
      data?.detail ||
      data?.message ||
      "Não foi possível concluir a operação.";

    const error = new Error(message);
    error.status = response.status;
    throw error;
  }

  return data;
}


function showPasswordChange() {
  loginView.hidden = true;
  changePasswordView.hidden = false;

  currentPasswordInput.value = temporaryPassword;
  currentPasswordInput.focus();
}


async function checkExistingSession() {
  try {
    const data = await requestJson("/auth/me");

    if (data?.user?.must_change_password) {
      loginView.hidden = true;
      changePasswordView.hidden = false;
      currentPasswordInput.focus();
      return;
    }

    window.location.replace("/");
  } catch (error) {
    if (error.status !== 401) {
      console.error("Falha ao verificar sessão:", error);
    }
  }
}


loginForm.addEventListener("submit", async event => {
  event.preventDefault();
  setMessage(loginMessage);

  const email = emailInput.value.trim().toLowerCase();
  const password = passwordInput.value;

  if (!email || !password) {
    setMessage(
      loginMessage,
      "Informe o e-mail e a senha."
    );
    return;
  }

  setButtonLoading(loginButton, true);

  try {
    const data = await requestJson("/auth/login", {
      method: "POST",
      body: JSON.stringify({
        email,
        password,
      }),
    });

    if (data.must_change_password) {
      temporaryPassword = password;
      passwordInput.value = "";
      showPasswordChange();
      return;
    }

    window.location.replace("/");
  } catch (error) {
    setMessage(
      loginMessage,
      error.message || "E-mail ou senha inválidos."
    );

    passwordInput.select();
  } finally {
    setButtonLoading(loginButton, false);
  }
});


changePasswordForm.addEventListener(
  "submit",
  async event => {
    event.preventDefault();
    setMessage(changePasswordMessage);

    const currentPassword =
      currentPasswordInput.value || temporaryPassword;

    const newPassword = newPasswordInput.value;
    const confirmPassword = confirmPasswordInput.value;

    if (!currentPassword || !newPassword || !confirmPassword) {
      setMessage(
        changePasswordMessage,
        "Preencha todos os campos."
      );
      return;
    }

    if (newPassword.length < 10) {
      setMessage(
        changePasswordMessage,
        "A nova senha deve ter pelo menos 10 caracteres."
      );
      return;
    }

    if (newPassword !== confirmPassword) {
      setMessage(
        changePasswordMessage,
        "A confirmação não corresponde à nova senha."
      );

      confirmPasswordInput.select();
      return;
    }

    if (newPassword === currentPassword) {
      setMessage(
        changePasswordMessage,
        "A nova senha deve ser diferente da senha temporária."
      );
      return;
    }

    setButtonLoading(changePasswordButton, true);

    try {
      await requestJson("/auth/change-password", {
        method: "POST",
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword,
          confirm_password: confirmPassword,
        }),
      });

      setMessage(
        changePasswordMessage,
        "Senha alterada. Redirecionando para o login…",
        "success"
      );

      window.setTimeout(() => {
        window.location.replace("/login");
      }, 900);
    } catch (error) {
      if (error.status === 401) {
        setMessage(
          changePasswordMessage,
          "Sua sessão expirou. Entre novamente."
        );

        window.setTimeout(() => {
          window.location.replace("/login");
        }, 1200);

        return;
      }

      setMessage(
        changePasswordMessage,
        error.message || "Não foi possível alterar a senha."
      );
    } finally {
      setButtonLoading(changePasswordButton, false);
    }
  }
);


document
  .querySelectorAll("[data-toggle-password]")
  .forEach(button => {
    button.addEventListener("click", () => {
      const inputId = button.dataset.togglePassword;
      const input = document.getElementById(inputId);

      if (!input) {
        return;
      }

      const showing = input.type === "text";

      input.type = showing ? "password" : "text";
      button.textContent = showing ? "Mostrar" : "Ocultar";
      button.setAttribute(
        "aria-label",
        showing ? "Mostrar senha" : "Ocultar senha"
      );
    });
  });


checkExistingSession();
