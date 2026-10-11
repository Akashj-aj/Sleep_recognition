document.addEventListener("DOMContentLoaded", () => {
    // ------------------------------------------------------------------
    // 0. Persistent Theme Toggle
    // ------------------------------------------------------------------
    const themeToggle = document.getElementById("theme-toggle");
    const savedTheme = localStorage.getItem("sleep-theme");

    const setTheme = (theme) => {
        const isLight = theme === "light";
        document.documentElement.dataset.theme = isLight ? "light" : "dark";
        if (themeToggle) {
            themeToggle.setAttribute("aria-pressed", String(isLight));
            themeToggle.setAttribute("aria-label", isLight ? "Switch to dark theme" : "Switch to light theme");
            themeToggle.querySelector(".theme-toggle-icon").textContent = isLight ? "☾" : "☼";
            themeToggle.querySelector(".theme-toggle-label").textContent = isLight ? "Dark" : "Light";
        }
    };

    setTheme(savedTheme === "light" ? "light" : "dark");
    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const nextTheme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
            localStorage.setItem("sleep-theme", nextTheme);
            setTheme(nextTheme);
        });
    }

    // ------------------------------------------------------------------
    // 1. Scroll-Driven Reveal Animations (IntersectionObserver)
    //    Fade-and-slide: elements start transparent + 60px lower,
    //    move into place over 0.6s with sequenced delays.
    // ------------------------------------------------------------------
    const revealElements = document.querySelectorAll(".reveal, .reveal-scale");

    if (revealElements.length > 0) {
        const revealObserver = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("revealed");
                        revealObserver.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
        );

        revealElements.forEach((el) => revealObserver.observe(el));
    }

    // ------------------------------------------------------------------
    // 2. Auto-tag elements for reveal animation if not already tagged
    //    This applies reveal classes to key structural elements
    // ------------------------------------------------------------------
    const autoRevealSelectors = [
        ".page-intro-section",
        ".stat-card",
        ".card",
        ".dashboard-header",
        ".limitation-banner",
        ".preset-bar",
        ".disclaimer-banner",
        ".best-model-card"
    ];

    let globalDelay = 0;
    autoRevealSelectors.forEach((selector) => {
        document.querySelectorAll(selector).forEach((el) => {
            if (!el.classList.contains("reveal") && !el.classList.contains("reveal-scale")) {
                el.classList.add("reveal");
                // Apply sequenced delays (0.2s increments, capped at delay-8)
                const delayClass = `delay-${Math.min(globalDelay + 1, 8)}`;
                el.classList.add(delayClass);
                globalDelay = (globalDelay + 1) % 8;

                // Observe it
                const obs = new IntersectionObserver(
                    (entries) => {
                        entries.forEach((entry) => {
                            if (entry.isIntersecting) {
                                entry.target.classList.add("revealed");
                                obs.unobserve(entry.target);
                            }
                        });
                    },
                    { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
                );
                obs.observe(el);
            }
        });
    });

    // ------------------------------------------------------------------
    // 3. Navbar Scroll State Change
    //    Add a subtle border highlight when scrolled past 50px
    // ------------------------------------------------------------------
    const navWrapper = document.querySelector(".top-nav-wrapper");
    if (navWrapper) {
        const onScroll = () => {
            navWrapper.classList.toggle("scrolled", window.scrollY > 50);
        };
        window.addEventListener("scroll", onScroll, { passive: true });
        onScroll();
    }

    // ------------------------------------------------------------------
    // 4. Mobile Navigation Toggle
    // ------------------------------------------------------------------
    const menuBtn = document.getElementById("mobile-menu-btn");
    const navMenu = document.getElementById("nav-menu");

    if (menuBtn && navMenu) {
        menuBtn.addEventListener("click", () => {
            navMenu.classList.toggle("open");
        });
    }

    // ------------------------------------------------------------------
    // 5. Preset Sample Profile Populator
    // ------------------------------------------------------------------
    const presets = {
        A: {
            age: 25,
            gender: "Male",
            occupation: "Software Engineer",
            sleep_duration: 7.5,
            physical_activity: 45,
            stress_level: 3,
            bmi_category: "Normal",
            sleep_disorder: "None",
            systolic_bp: 120,
            diastolic_bp: 80,
            heart_rate: 72,
            daily_steps: 7000,
            model: "random_forest"
        },
        B: {
            age: 42,
            gender: "Female",
            occupation: "Nurse",
            sleep_duration: 6.1,
            physical_activity: 30,
            stress_level: 8,
            bmi_category: "Overweight",
            sleep_disorder: "Insomnia",
            systolic_bp: 135,
            diastolic_bp: 88,
            heart_rate: 78,
            daily_steps: 4500,
            model: "logistic_regression"
        },
        C: {
            age: 50,
            gender: "Male",
            occupation: "Doctor",
            sleep_duration: 5.9,
            physical_activity: 60,
            stress_level: 7,
            bmi_category: "Obese",
            sleep_disorder: "Sleep Apnea",
            systolic_bp: 140,
            diastolic_bp: 90,
            heart_rate: 85,
            daily_steps: 3000,
            model: "svm"
        }
    };

    document.querySelectorAll("[data-preset]").forEach((button) => {
        button.addEventListener("click", () => {
            const key = button.getAttribute("data-preset");
            const profile = presets[key];
            if (profile) {
                Object.entries(profile).forEach(([fieldId, value]) => {
                    const field = document.getElementById(fieldId);
                    if (field) {
                        field.value = value;
                    }
                });
                // Clear inline error text
                document.querySelectorAll(".field-error").forEach((el) => el.textContent = "");
            }
        });
    });

    // ------------------------------------------------------------------
    // 6. Clear / Reset Form Handler
    // ------------------------------------------------------------------
    const resetBtn = document.getElementById("reset-form-btn");
    const predictionForm = document.getElementById("prediction-form");

    if (resetBtn && predictionForm) {
        resetBtn.addEventListener("click", () => {
            predictionForm.reset();
            document.querySelectorAll(".field-error").forEach((el) => el.textContent = "");
            const resultDisplay = document.getElementById("result-display");
            if (resultDisplay) {
                resultDisplay.style.opacity = "0.5";
            }
        });
    }

    // ------------------------------------------------------------------
    // 7. Live Table Search Filter (Dataset Page)
    // ------------------------------------------------------------------
    const searchInput = document.getElementById("dataset-search");
    const datasetTable = document.getElementById("dataset-table");

    if (searchInput && datasetTable) {
        searchInput.addEventListener("input", () => {
            const query = searchInput.value.toLowerCase().trim();
            const rows = datasetTable.querySelectorAll("tbody tr");

            rows.forEach((row) => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(query) ? "" : "none";
            });
        });
    }

    // ------------------------------------------------------------------
    // 8. Client-Side Form Validation
    // ------------------------------------------------------------------
    if (predictionForm) {
        predictionForm.addEventListener("submit", (e) => {
            let isValid = true;

            const clearErrors = () => {
                document.querySelectorAll(".field-error").forEach((el) => el.textContent = "");
            };
            clearErrors();

            const setError = (fieldId, message) => {
                const errSpan = document.getElementById(`${fieldId}-error`);
                if (errSpan) {
                    errSpan.textContent = message;
                }
                isValid = false;
            };

            const age = Number(document.getElementById("age").value);
            const sleepDuration = Number(document.getElementById("sleep_duration").value);
            const physicalActivity = Number(document.getElementById("physical_activity").value);
            const stressLevel = Number(document.getElementById("stress_level").value);
            const systolicBp = Number(document.getElementById("systolic_bp").value);
            const diastolicBp = Number(document.getElementById("diastolic_bp").value);
            const heartRate = Number(document.getElementById("heart_rate").value);
            const dailySteps = Number(document.getElementById("daily_steps").value);

            const gender = document.getElementById("gender").value;
            const occupation = document.getElementById("occupation").value;
            const bmiCategory = document.getElementById("bmi_category").value;
            const sleepDisorder = document.getElementById("sleep_disorder").value;

            if (!age || age < 1 || age > 120 || !Number.isInteger(age)) {
                setError("age", "Enter a valid age between 1 and 120 years.");
            }
            if (!gender) {
                setError("gender", "Select a valid gender.");
            }
            if (!occupation) {
                setError("occupation", "Select a valid occupation.");
            }
            if (!sleepDuration || sleepDuration <= 0 || sleepDuration > 24) {
                setError("sleep_duration", "Enter sleep duration between 0.1 and 24 hours.");
            }
            if (isNaN(physicalActivity) || physicalActivity < 0) {
                setError("physical_activity", "Physical activity duration cannot be negative.");
            }
            if (!stressLevel || stressLevel < 1 || stressLevel > 10 || !Number.isInteger(stressLevel)) {
                setError("stress_level", "Select a stress level scale from 1 to 10.");
            }
            if (isNaN(dailySteps) || dailySteps < 0) {
                setError("daily_steps", "Daily steps must be a non-negative number.");
            }
            if (!bmiCategory) {
                setError("bmi_category", "Select a valid BMI category.");
            }
            if (!sleepDisorder) {
                setError("sleep_disorder", "Select a valid sleep disorder option.");
            }
            if (!systolicBp || systolicBp < 50 || systolicBp > 300) {
                setError("systolic_bp", "Upper blood pressure must be between 50 and 300 mmHg.");
            }
            if (!diastolicBp || diastolicBp < 30 || diastolicBp > 200) {
                setError("diastolic_bp", "Lower blood pressure must be between 30 and 200 mmHg.");
            }
            if (!heartRate || heartRate < 1 || heartRate > 250) {
                setError("heart_rate", "Heart rate must be between 1 and 250 beats/min.");
            }

            if (!isValid) {
                e.preventDefault();
                const firstError = document.querySelector(".field-error:not(:empty)");
                if (firstError) {
                    firstError.parentElement.scrollIntoView({ behavior: "smooth", block: "center" });
                }
            }
        });
    }
});
