(() => {
  let workspaceRevision = 0;
  let popstateBound = false;
  let activeStream = null;

  const parsePayload = async (response) => {
    try {
      return await response.json();
    } catch (_error) {
      return {};
    }
  };

  const responseMessage = (payload, fallback) => payload?.message || payload?.detail || fallback;
  const setText = (element, text, hidden = false) => {
    if (!element) return;
    element.textContent = text;
    element.hidden = hidden;
  };
  const setDisabled = (controls, disabled) => {
    controls.forEach((control) => {
      if (disabled) {
        control.dataset.wasDisabled = control.disabled ? "true" : "false";
        control.disabled = true;
      } else if (control.dataset.wasDisabled !== "true") {
        control.disabled = false;
        delete control.dataset.wasDisabled;
      }
    });
  };
  const currentShell = () => document.querySelector("main.page-shell");
  const isCurrent = (revision) => revision === workspaceRevision;
  const canIntercept = (event) => (
    event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey
  );
  // A finished background job should update the current card quietly. Manual
  // wizard navigation remains animated and scrolls to the selected card.
  const backgroundRefreshOptions = Object.freeze({ animate: false, preserveScroll: true, scroll: false });

  const activateMarkdownTab = (tabs, selectedName) => {
    if (!tabs) return;
    tabs.querySelectorAll("[data-markdown-tab]").forEach((tab) => {
      const selected = tab.dataset.markdownTab === selectedName;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    tabs.querySelectorAll("[data-markdown-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.markdownPanel !== selectedName;
    });
  };

  const closeActiveStream = (root = null) => {
    if (!activeStream || (root && activeStream.root !== root)) return;
    activeStream.source.close();
    activeStream = null;
  };

  const replaceWorkspace = (current, next, { animate = true } = {}) => {
    // A content stream belongs to its current card. Never allow an old
    // EventSource to complete after the user has navigated elsewhere.
    closeActiveStream(current);
    const replace = () => current.replaceWith(next);
    if (animate && typeof document.startViewTransition === "function") {
      document.startViewTransition(replace);
    } else {
      replace();
    }
  };

  const scrollToRequestedStage = (url) => {
    const stageId = url.hash.slice(1);
    if (!stageId) return;
    window.requestAnimationFrame(() => {
      document.getElementById(stageId)?.scrollIntoView({ block: "start" });
    });
  };

  // The server remains the source of truth for which wizard card is reachable.
  // Fetching and replacing only the workspace prevents stale, disabled controls
  // after a background job while keeping ordinary form redirects as a no-JS fallback.
  const loadWorkspace = async (
    target,
    historyMode = "push",
    { animate = true, preserveScroll = false, scroll = true } = {},
  ) => {
    const targetUrl = new URL(target, window.location.href);
    if (targetUrl.origin !== window.location.origin) {
      window.location.assign(targetUrl.toString());
      return;
    }
    const revision = ++workspaceRevision;
    try {
      const response = await window.fetch(targetUrl, { headers: { Accept: "text/html" }, cache: "no-store" });
      if (!response.ok) throw new Error(`页面刷新失败（${response.status}）`);
      const documentText = await response.text();
      if (!isCurrent(revision)) return;
      const nextDocument = new DOMParser().parseFromString(documentText, "text/html");
      const next = nextDocument.querySelector("main.page-shell");
      const current = currentShell();
      if (!next || !current) throw new Error("页面结构不完整");

      const finalUrl = new URL(response.url || targetUrl, window.location.href);
      // Fetch responses omit fragments. Preserve the requested export/step
      // anchor when the server has not redirected to another wizard card.
      if (!finalUrl.hash && finalUrl.pathname === targetUrl.pathname && finalUrl.search === targetUrl.search) {
        finalUrl.hash = targetUrl.hash;
      }
      const scrollPosition = preserveScroll ? { x: window.scrollX, y: window.scrollY } : null;
      replaceWorkspace(current, next, { animate });
      if (historyMode === "push") window.history.pushState({}, "", finalUrl);
      if (historyMode === "replace") window.history.replaceState({}, "", finalUrl);
      initialize(next, revision);
      if (scroll) {
        scrollToRequestedStage(finalUrl);
      } else if (scrollPosition) {
        window.requestAnimationFrame(() => window.scrollTo(scrollPosition.x, scrollPosition.y));
      }
    } catch (_error) {
      // Preserve a reliable browser navigation path if a proxy or a template
      // response cannot be upgraded in place.
      window.location.assign(targetUrl.toString());
    }
  };

  const loadWizardStep = (step, historyMode = "push", options = {}) => {
    const url = new URL(window.location.href);
    url.searchParams.set("step", step);
    url.hash = `${step}-stage`;
    return loadWorkspace(url, historyMode, options);
  };

  const initialize = (root, revision = workspaceRevision) => {
    const selectAll = (selector) => Array.from(root.querySelectorAll(selector));
    const jobPanelFor = (element) => element.closest("[data-job-status-url]") || root.querySelector("[data-job-status-url]");
    const jobControls = () => selectAll(
      "form[data-job-form] button, form[data-job-form] input, form[data-job-form] select, "
      + "form[data-title-confirm-form] button, form[data-title-confirm-form] input, "
      + "form[data-outline-form] button, form[data-outline-form] input, form[data-outline-form] textarea, "
      + "form[data-outline-confirm-form] button, form[data-outline-confirm-form] input, "
      + "form[data-content-form] button, form[data-content-form] input, form[data-content-form] textarea, "
      + "form[data-image-plan-form] button, form[data-image-plan-form] input, form[data-image-plan-form] select, form[data-image-plan-form] textarea, "
      + "form[data-pending-form] button, form[data-pending-form] input, form[data-pending-form] select, form[data-pending-form] textarea"
    );
    const showJobError = (panel, message) => setText(panel?.querySelector("[data-job-error]"), message, false);

    selectAll("[data-markdown-tabs]").forEach((tabs) => {
      const tabButtons = Array.from(tabs.querySelectorAll("[data-markdown-tab]"));
      tabButtons.forEach((tab, index) => {
        tab.addEventListener("click", () => activateMarkdownTab(tabs, tab.dataset.markdownTab));
        tab.addEventListener("keydown", (event) => {
          const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
          if (!keys.includes(event.key)) return;
          event.preventDefault();
          const nextIndex = event.key === "Home"
            ? 0
            : event.key === "End"
              ? tabButtons.length - 1
              : (index + (event.key === "ArrowRight" ? 1 : -1) + tabButtons.length) % tabButtons.length;
          const nextTab = tabButtons[nextIndex];
          activateMarkdownTab(tabs, nextTab.dataset.markdownTab);
          nextTab.focus();
        });
      });
    });

    selectAll("[data-wizard-progress] a, .wizard-card-navigation a").forEach((link) => {
      link.addEventListener("click", (event) => {
        if (!canIntercept(event)) return;
        if (link.getAttribute("href")?.startsWith("#")) return;
        event.preventDefault();
        void loadWorkspace(link.href, "push");
      });
    });

    const renderJobStatus = (panel, job) => {
      const progress = panel?.querySelector("[data-job-progress]");
      const statusBadge = panel?.querySelector(".status");
      const elapsed = Number.isInteger(job?.elapsedSeconds) ? ` · 已耗时 ${job.elapsedSeconds}s` : "";
      const action = job?.message || job?.action || "后台任务";
      const labels = {
        QUEUED: "排队中",
        RUNNING: "生成中",
        SUCCESS: "已完成",
        FAILED: "需要处理",
        CANCELLING: "正在取消",
      };
      const status = String(job?.status || "").toUpperCase();
      if (statusBadge && job?.status) {
        statusBadge.textContent = labels[status] || job.status;
        statusBadge.className = `status ${status.toLowerCase()}`;
      }
      const progressLabel = status === "QUEUED" ? "排队中" : status === "CANCELLING" ? "正在取消" : "正在执行";
      setText(progress, `${progressLabel}：${action}${elapsed}`, !["QUEUED", "RUNNING", "CANCELLING"].includes(status));
    };

    const canonicalDetailUrl = (panel, job) => {
      if (panel?.dataset.jobDetailUrl) return panel.dataset.jobDetailUrl;
      if (job?.taskId) return `/article/detail/${job.taskId}`;
      return window.location.pathname;
    };

    const workflowStepForAction = (action) => {
      const normalized = String(action || "").split(":", 1)[0];
      const steps = {
        GENERATE_TITLES: "title",
        WAIT_FOR_TITLE: "title",
        GENERATE_OUTLINE: "outline",
        GENERATE_CONTENT: "content",
        REGENERATE_CONTENT: "content",
        GENERATE_IMAGE_PROMPTS: "images",
        REGENERATE_IMAGE_PROMPTS: "images",
        GENERATE_IMAGE_RESULTS: "images",
        REGENERATE_IMAGE_RESULTS: "images",
        REGENERATE_IMAGE_RESULT: "images",
        COMPOSE_ARTICLE: "images",
        REVIEW_ARTICLE: "delivery",
        MANUAL_REVIEW: "delivery",
        COMPLETE: "delivery",
      };
      return steps[normalized] || null;
    };

    const activeWizardStep = () => root.querySelector("[data-wizard-workspace]")?.dataset.activeStep || null;

    const pollJob = async (panel, attempt = 0) => {
      if (!isCurrent(revision) || !panel?.isConnected) return;
      const statusUrl = panel.dataset.jobStatusUrl;
      if (!statusUrl) return;
      try {
        const response = await window.fetch(statusUrl, { headers: { Accept: "application/json" }, cache: "no-store" });
        const payload = await parsePayload(response);
        if (!response.ok) {
          if ([400, 404].includes(response.status)) {
            window.location.assign("/article/list");
            return;
          }
          showJobError(panel, responseMessage(payload, "无法读取任务状态，请稍后刷新页面。"));
          window.setTimeout(() => void pollJob(panel, attempt + 1), Math.min(5000, 1000 * (attempt + 1)));
          return;
        }
        const job = payload.data || {};
        renderJobStatus(panel, job);
        const status = String(job.status || "").toUpperCase();
        if (["QUEUED", "RUNNING", "CANCELLING"].includes(status)) {
          // A one-click job has no user confirmation checkpoints.  Keep the
          // visible wizard card in sync with its persisted concrete substep,
          // rather than leaving the user on the title card until everything
          // suddenly appears complete.
          const targetStep = job.executionMode === "ONE_CLICK" ? workflowStepForAction(job.action) : null;
          const activeStep = activeWizardStep();
          if (targetStep && targetStep !== activeStep) {
            await loadWizardStep(targetStep, "replace", backgroundRefreshOptions);
            return;
          }
          window.setTimeout(() => void pollJob(panel), 1000);
          return;
        }
        if (status === "SUCCESS") {
          await loadWorkspace(canonicalDetailUrl(panel, job), "replace", backgroundRefreshOptions);
          return;
        }
        if (status === "FAILED") {
          if (job.errorMessage) showJobError(panel, job.errorMessage);
          await loadWorkspace(canonicalDetailUrl(panel, job), "replace", backgroundRefreshOptions);
          return;
        }
        setDisabled(jobControls(), false);
      } catch (_error) {
        showJobError(panel, "网络连接失败，任务可能仍在后台执行；请稍后刷新页面查看状态。");
        window.setTimeout(() => void pollJob(panel, attempt + 1), Math.min(5000, 1000 * (attempt + 1)));
      }
    };

    const startJob = async (form) => {
      const warning = form.dataset.confirm;
      if (warning && !window.confirm(warning)) return;
      const panel = jobPanelFor(form);
      const button = form.querySelector("button[type='submit']");
      const url = form.dataset.jobUrl;
      if (!url) return;
      const formData = new FormData(form);
      setText(panel?.querySelector("[data-job-error]"), "", true);
      setDisabled(jobControls(), true);
      if (button) {
        button.dataset.originalText = button.textContent;
        button.textContent = form.dataset.pendingText || "任务已提交…";
      }
      try {
        const response = await window.fetch(url, { method: "POST", body: formData, headers: { Accept: "application/json" } });
        const payload = await parsePayload(response);
        if (!response.ok) {
          showJobError(panel, responseMessage(payload, `任务提交失败（${response.status}）。`));
          setDisabled(jobControls(), false);
          if (button?.dataset.originalText) button.textContent = button.dataset.originalText;
          return;
        }
        renderJobStatus(panel, payload.data || {});
        void pollJob(panel);
      } catch (_error) {
        showJobError(panel, "网络连接失败，任务没有提交；请检查服务后重试。");
        setDisabled(jobControls(), false);
        if (button?.dataset.originalText) button.textContent = button.dataset.originalText;
      }
    };

    const submitWizardForm = async (form, event) => {
      event.preventDefault();
      const warning = form.dataset.confirm;
      if (warning && !window.confirm(warning)) return;
      const submitter = event.submitter || form.querySelector("button[type='submit']");
      const formData = new FormData(form);
      if (submitter) {
        submitter.dataset.originalText = submitter.textContent;
        submitter.textContent = form.dataset.pendingText || "处理中…";
        submitter.disabled = true;
      }
      try {
        const response = await window.fetch(form.action, { method: "POST", body: formData, redirect: "follow" });
        if (!response.ok) {
          const payload = await parsePayload(response);
          throw new Error(responseMessage(payload, `保存失败（${response.status}）。`));
        }
        await loadWorkspace(response.url, "replace");
      } catch (error) {
        const target = form.querySelector("[data-form-error]") || root.querySelector("[data-job-error]");
        setText(target, error.message || "保存失败，请重试。", false);
        if (submitter) {
          submitter.textContent = submitter.dataset.originalText || "保存";
          submitter.disabled = false;
        }
      }
    };

    selectAll("form[data-pending-form]").forEach((form) => {
      form.addEventListener("submit", (event) => {
        if (form.matches("[data-title-confirm-form], [data-job-form], [data-create-form], [data-outline-form], [data-outline-confirm-form], [data-content-form]")) return;
        const warning = form.dataset.confirm;
        if (warning && !window.confirm(warning)) {
          event.preventDefault();
          return;
        }
        const submitter = event.submitter;
        if (submitter) {
          submitter.dataset.originalText = submitter.textContent;
          submitter.textContent = form.dataset.pendingText || "处理中…";
          submitter.disabled = true;
        }
      });
    });

    selectAll("form[data-outline-form], form[data-outline-confirm-form], form[data-content-form]").forEach((form) => {
      form.addEventListener("submit", (event) => void submitWizardForm(form, event));
    });

    selectAll("form[data-title-confirm-form]").forEach((form) => {
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const warning = form.dataset.confirm;
        if (warning && !window.confirm(warning)) return;
        const progress = root.querySelector("[data-title-progress]");
        const error = root.querySelector("[data-title-error]");
        const controls = Array.from(root.querySelectorAll("form[data-title-confirm-form] button, form[data-title-confirm-form] input"));
        const formData = new FormData(form);
        setText(error, "", true);
        setText(progress, "正在保存标题…", false);
        setDisabled(controls, true);
        try {
          const response = await window.fetch(form.action, { method: "POST", body: formData, headers: { Accept: "application/json" } });
          const payload = await parsePayload(response);
          if (!response.ok) throw new Error(responseMessage(payload, `标题保存失败（${response.status}）。`));
          await loadWorkspace(response.url, "replace");
        } catch (requestError) {
          setText(error, requestError.message || "标题保存失败，请重试。", false);
          setText(progress, "", true);
          setDisabled(controls, false);
        }
      });
    });

    selectAll("form[data-job-form]").forEach((form) => {
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        void startJob(form);
      });
    });

    selectAll("form[data-create-form]").forEach((form) => {
      form.addEventListener("submit", async (event) => {
        const submitter = event.submitter;
        if (submitter?.value !== "one_click") return;
        event.preventDefault();
        const error = form.querySelector("[data-create-error]");
        const controls = Array.from(form.querySelectorAll("button, input, select, textarea"));
        const formData = new FormData(form);
        setText(error, "", true);
          setDisabled(controls, true);
        try {
          formData.set("mode", "one_click");
          // One-click work continues in the background. Return to the list so
          // the author can see its stage without being forced into a card.
          const response = await window.fetch(form.action, {
            method: "POST",
            body: formData,
            headers: { Accept: "application/json" },
          });
          const payload = await parsePayload(response);
          if (!response.ok) throw new Error(responseMessage(payload, `创建任务失败（${response.status}）。`));
          const listUrl = payload?.data?.listUrl;
          if (!listUrl) throw new Error("任务已创建，但未收到任务列表地址。请刷新后继续。");
          window.location.assign(listUrl);
        } catch (requestError) {
          setText(error, requestError.message || "创建任务失败，请重试。", false);
          setDisabled(controls, false);
        }
      });
    });

    selectAll("form[data-delete-form]").forEach((form) => {
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!window.confirm(form.dataset.confirm || "确认删除这篇任务吗？")) return;
        const card = form.closest("[data-task-card]");
        const error = card?.querySelector("[data-delete-error]");
        const controls = Array.from(form.querySelectorAll("button, input"));
        const button = event.submitter || form.querySelector("button[type='submit']");
        const formData = new FormData(form);
        setText(error, "", true);
        if (button) {
          button.dataset.originalText = button.textContent;
          button.textContent = ["queued", "running"].includes(card?.dataset.taskState) ? "正在取消并删除…" : "删除中…";
        }
        setDisabled(controls, true);
        try {
          const response = await window.fetch(form.action, { method: "POST", body: formData, headers: { Accept: "application/json" } });
          const payload = await parsePayload(response);
          if (!response.ok || payload?.data !== true) throw new Error(responseMessage(payload, `删除任务失败（${response.status}）。`));
          window.location.reload();
        } catch (requestError) {
          setText(error, requestError.message || "删除任务失败，请重试。", false);
          setDisabled(controls, false);
          if (button?.dataset.originalText) button.textContent = button.dataset.originalText;
        }
      });
    });

    const taskList = root.querySelector("[data-task-list]");
    if (taskList?.dataset.taskStatusUrl) {
      const updateTaskCard = (task) => {
        const card = taskList.querySelector(`[data-task-id="${task.taskId}"]`);
        if (!card) return;
        card.dataset.taskState = task.state;
        const status = card.querySelector("[data-task-status]");
        if (status) {
          status.className = `task-status ${task.tone}`;
          status.setAttribute("aria-label", `任务状态：${task.label}`);
        }
        const icon = card.querySelector("[data-task-status-icon]");
        if (icon) {
          icon.textContent = task.icon;
          icon.className = `task-status-icon${["queued", "running"].includes(task.state) ? " spinner" : ""}`;
        }
        setText(card.querySelector("[data-task-status-label]"), task.label, false);
        setText(card.querySelector("[data-task-progress]"), task.progress || "", !task.progress);
      };
      const pollTaskList = async () => {
        if (!isCurrent(revision) || !taskList.isConnected) return;
        try {
          const response = await window.fetch(taskList.dataset.taskStatusUrl, {
            headers: { Accept: "application/json" },
            cache: "no-store",
          });
          const payload = await parsePayload(response);
          if (response.ok && Array.isArray(payload?.data)) payload.data.forEach(updateTaskCard);
        } catch (_error) {
          // Keep local task cards usable when the service is briefly restarting.
        }
        window.setTimeout(() => void pollTaskList(), 1200);
      };
      void pollTaskList();
    }

    const cancellingCard = root.querySelector('[data-task-state="cancelling"]');
    if (cancellingCard?.dataset.taskId) {
      const pollCancellation = async () => {
        if (!isCurrent(revision) || !cancellingCard.isConnected) return;
        try {
          const response = await window.fetch(`/article/${cancellingCard.dataset.taskId}/jobs/status`, {
            headers: { Accept: "application/json" },
            cache: "no-store",
          });
          if (!response.ok) {
            await loadWorkspace("/article/list", "replace");
            return;
          }
        } catch (_error) {
          // Keep the card visible while the local service is temporarily unavailable.
        }
        window.setTimeout(() => void pollCancellation(), 1000);
      };
      void pollCancellation();
    }

    selectAll("form[data-outline-form]").forEach((form) => {
      const editor = form.querySelector("[data-outline-editor]");
      const saveButton = form.querySelector("[data-outline-save]");
      const status = form.querySelector("[data-outline-status]");
      if (!editor || !saveButton) return;
      const initialValue = editor.value;
      const requiresConfirmation = editor.dataset.outlineConfirmation === "required";
      const sync = () => {
        const shouldSave = requiresConfirmation || editor.value !== initialValue;
        saveButton.hidden = !shouldSave;
        saveButton.disabled = !shouldSave;
        if (!requiresConfirmation && status) {
          status.textContent = shouldSave ? "大纲已修改，保存后会重新生成下游内容。" : "大纲已保存；修改内容后才需要再次保存。";
        }
      };
      editor.addEventListener("input", sync);
      sync();
    });

    selectAll("[data-image-count-mode]").forEach((select) => {
      const field = select.closest("form")?.querySelector("[data-image-count-field]");
      const input = field?.querySelector("input");
      if (!field || !input) return;
      const sync = () => {
        const custom = select.value === "CUSTOM";
        field.hidden = !custom;
        input.disabled = !custom;
        if (!custom) input.value = "";
      };
      select.addEventListener("change", sync);
      sync();
    });

    const storyboard = root.querySelector("[data-image-storyboard]");
    if (storyboard) {
      let draggedCard = null;
      let reorderPending = false;
      let anchorParagraphs = {};
      try {
        anchorParagraphs = JSON.parse(storyboard.dataset.anchorParagraphs || "{}");
      } catch (_error) {
        anchorParagraphs = {};
      }
      const cards = () => Array.from(storyboard.querySelectorAll("[data-image-plan-card]"));
      const isFixedCard = (card) => card?.dataset.imageFixed === "true";
      const planError = (form, message, hidden = false) => setText(form.querySelector("[data-image-plan-error]"), message, hidden);
      const serializePlan = (form) => Object.fromEntries(new FormData(form).entries());
      const endpointError = (payload, fallback) => payload?.detail || responseMessage(payload, fallback);

      const syncAnchorControls = (form, resetParagraph = false) => {
        const placement = form.querySelector("[data-image-placement]");
        const heading = form.querySelector("[data-image-anchor-heading]");
        const paragraph = form.querySelector("[data-image-anchor-paragraph]");
        if (!placement || !heading || !paragraph) return;
        const isGlobal = ["COVER", "AFTER_INTRO"].includes(placement.value);
        if (isGlobal) {
          heading.value = "";
          paragraph.innerHTML = '<option value="0">不需要段落锚点</option>';
          paragraph.value = "0";
          heading.disabled = true;
          paragraph.disabled = true;
          return;
        }
        heading.disabled = false;
        paragraph.disabled = false;
        const selected = resetParagraph ? "0" : paragraph.value;
        const choices = anchorParagraphs[heading.value] || [];
        paragraph.replaceChildren();
        const sectionEnd = new Option("章节末尾", "0", false, selected === "0");
        paragraph.add(sectionEnd);
        choices.forEach((number) => paragraph.add(new Option(`第 ${number} 段之后`, String(number), false, String(number) === selected)));
        if (placement.value === "AFTER_PARAGRAPH" && selected === "0" && choices.length) paragraph.value = String(choices[0]);
      };

      const updateVisualChangeHint = (form) => {
        const original = form.dataset.visualFingerprint || "";
        const current = Array.from(form.querySelectorAll("[data-image-visual-field]")).map((field) => field.value).join("\u0000");
        if (original && original !== current) {
          planError(form, "画面内容已调整：保存后会清空当前图片，需重新生成。", false);
        } else {
          planError(form, "", true);
        }
      };

      const persistOrder = async () => {
        if (reorderPending) return;
        const order = cards().map((card) => Number(card.dataset.index));
        const url = storyboard.dataset.reorderUrl;
        if (!url || order.some((index) => !Number.isInteger(index))) return;
        reorderPending = true;
        storyboard.classList.add("is-saving");
        try {
          const response = await window.fetch(url, {
            method: "POST",
            headers: { Accept: "application/json", "Content-Type": "application/json" },
            body: JSON.stringify({ indices: order }),
          });
          const payload = await parsePayload(response);
          if (!response.ok) throw new Error(endpointError(payload, "配图排序保存失败。"));
          await loadWizardStep("images", "replace");
        } catch (error) {
          const firstForm = storyboard.querySelector("[data-image-plan-form]");
          planError(firstForm, error.message || "配图排序保存失败，请刷新后重试。", false);
        } finally {
          reorderPending = false;
          storyboard.classList.remove("is-saving");
        }
      };

      selectAll("[data-image-plan-form]").forEach((form) => {
        const visualFields = Array.from(form.querySelectorAll("[data-image-visual-field]"));
        form.dataset.visualFingerprint = visualFields.map((field) => field.value).join("\u0000");
        syncAnchorControls(form);
        form.querySelector("[data-image-placement]")?.addEventListener("change", () => syncAnchorControls(form, true));
        form.querySelector("[data-image-anchor-heading]")?.addEventListener("change", () => syncAnchorControls(form, true));
        visualFields.forEach((field) => field.addEventListener("input", () => updateVisualChangeHint(form)));
        visualFields.forEach((field) => field.addEventListener("change", () => updateVisualChangeHint(form)));
        form.addEventListener("submit", async (event) => {
          event.preventDefault();
          const changedVisual = form.dataset.visualFingerprint !== Array.from(form.querySelectorAll("[data-image-visual-field]")).map((field) => field.value).join("\u0000");
          if (changedVisual && !window.confirm("修改画面内容会清空图片、图文稿与评审结果。确定保存吗？")) return;
          const button = event.submitter || form.querySelector("button[type='submit']");
          if (button) {
            button.dataset.originalText = button.textContent;
            button.textContent = "正在保存…";
            button.disabled = true;
          }
          planError(form, "", true);
          try {
            const response = await window.fetch(form.action, {
              method: "POST",
              headers: { Accept: "application/json", "Content-Type": "application/json" },
              body: JSON.stringify(serializePlan(form)),
            });
            const payload = await parsePayload(response);
            if (!response.ok) throw new Error(endpointError(payload, "图片编排保存失败。"));
            await loadWizardStep("images", "replace");
          } catch (error) {
            planError(form, error.message || "图片编排保存失败，请重试。", false);
            if (button) {
              button.textContent = button.dataset.originalText || "保存编排";
              button.disabled = false;
            }
          }
        });
      });

      cards().forEach((card) => {
        card.addEventListener("dragstart", (event) => {
          if (isFixedCard(card)) {
            event.preventDefault();
            return;
          }
          draggedCard = card;
          card.classList.add("is-dragging");
          event.dataTransfer.effectAllowed = "move";
        });
        card.addEventListener("dragend", () => {
          draggedCard = null;
          card.classList.remove("is-dragging");
        });
        card.addEventListener("dragover", (event) => {
          if (!draggedCard || draggedCard === card || isFixedCard(draggedCard) || isFixedCard(card)) return;
          event.preventDefault();
          event.dataTransfer.dropEffect = "move";
        });
        card.addEventListener("drop", (event) => {
          if (!draggedCard || draggedCard === card || isFixedCard(draggedCard) || isFixedCard(card)) return;
          event.preventDefault();
          const before = event.clientY < card.getBoundingClientRect().top + card.offsetHeight / 2;
          storyboard.insertBefore(draggedCard, before ? card : card.nextSibling);
          void persistOrder();
        });
        card.querySelectorAll("[data-image-move]").forEach((button) => {
          button.addEventListener("click", () => {
            const direction = Number(button.dataset.imageMove);
            const list = cards();
            const position = list.indexOf(card);
            const sibling = list[position + direction];
            if (!sibling || isFixedCard(card) || isFixedCard(sibling)) return;
            storyboard.insertBefore(card, direction < 0 ? sibling : sibling.nextSibling);
            void persistOrder();
          });
        });
      });
    }

    selectAll("[data-stream-url]").forEach((button) => {
      button.addEventListener("click", () => {
        const target = root.querySelector(button.dataset.contentTarget);
        const progress = root.querySelector(button.dataset.progressTarget);
        if (!target) return;
        const existingContent = target.value;
        if (existingContent.trim() && !window.confirm("重新生成正文会清空配图、图文稿和评审结果。确定继续吗？")) return;
        root.querySelector("#content-stage")?.scrollIntoView({ behavior: "smooth", block: "start" });
        activateMarkdownTab(target.closest("[data-markdown-tabs]"), "edit");
        let receivedContent = false;
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.textContent = "正在生成…";
        closeActiveStream();
        const source = new EventSource(button.dataset.streamUrl);
        activeStream = { source, root, revision };
        let settled = false;
        const streamIsCurrent = () => (
          !settled
          && activeStream?.source === source
          && activeStream?.root === root
          && isCurrent(revision)
          && root.isConnected
        );
        const restore = (message) => {
          if (settled) return;
          settled = true;
          if (!receivedContent) target.value = existingContent;
          if (progress) progress.textContent = message;
          button.textContent = button.dataset.originalText;
          button.disabled = false;
          source.close();
          if (activeStream?.source === source) activeStream = null;
        };
        source.addEventListener("progress", (event) => {
          if (!streamIsCurrent()) return;
          if (progress) progress.textContent = JSON.parse(event.data).message;
        });
        source.addEventListener("content", (event) => {
          if (!streamIsCurrent()) return;
          if (!receivedContent) { target.value = ""; receivedContent = true; }
          target.value += JSON.parse(event.data).content;
        });
        source.addEventListener("content_replace", (event) => {
          if (!streamIsCurrent()) return;
          target.value = JSON.parse(event.data).content;
          receivedContent = true;
        });
        source.addEventListener("done", () => {
          if (!streamIsCurrent()) return;
          settled = true;
          if (progress) progress.textContent = "正文已保存，正在进入配图步骤…";
          source.close();
          if (activeStream?.source === source) activeStream = null;
          window.setTimeout(() => void loadWizardStep("images", "replace", backgroundRefreshOptions), 200);
        });
        source.addEventListener("fail", (event) => {
          if (!streamIsCurrent()) return;
          restore(JSON.parse(event.data).message || "正文生成失败，已保留原正文。");
        });
        source.onerror = () => {
          if (streamIsCurrent()) restore("连接中断，未保存不完整内容；请重试。");
        };
      });
    });

    const panel = root.querySelector("[data-job-status-url]");
    if (panel?.dataset.jobActive === "true" || panel?.dataset.jobRunning === "true") {
      void pollJob(panel);
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    const root = currentShell();
    if (root) initialize(root, ++workspaceRevision);
    if (!popstateBound) {
      popstateBound = true;
      window.addEventListener("popstate", () => void loadWorkspace(window.location.href, "none"));
    }
  });
})();
